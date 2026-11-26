from settings import *
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
from scipy.special import expit
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

@dataclass
class Weights:
    early: np.ndarray
    late: np.ndarray
    tilt_bounded: int=0
    raw: object=None

def predict_probability(x,y,obs,C,x_predict=None):
    keep=obs>0
    require(set(y[keep])=={0,1},"Bootstrap fit has an empty treatment/censor class")
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        model=LogisticRegression(C=C,solver="lbfgs",max_iter=3500,tol=1e-9,random_state=SEED)
        model.fit(x[keep],y[keep],sample_weight=obs[keep])
    return np.clip(model.predict_proba(x if x_predict is None else x_predict)[:,1],1e-10,1-1e-10)

def weighted_quantile(x, counts, p):
    sorted_idx=np.argsort(x)
    values=x[sorted_idx]
    counts=counts[sorted_idx].astype(int)
    require(counts.sum()>1,"Too few terminal observations for the percentile truncation")
    return np.quantile(np.repeat(values,counts),p)

def estimate_weights(t,m):
    den0=predict_probability(t.x0,t.y0,m[t.i0],t.c)
    factor0=np.zeros((t.n,t.g))
    factor0[t.i0,t.h0]=-np.log1p(-den0)
    W0=np.exp(np.cumsum(factor0,axis=1))
    early=np.ones((2,t.n,t.g))
    early[0,:,1:]=W0[:,:-1]
    eligible=t.a1
    x=t.x[eligible]
    y=t.y1[eligible]
    if t.label=="S4":
        pid,hour,z,y_observed,z_g=t.xlib
        initiation=predict_probability(z,y_observed,m[pid],t.c,z_g).reshape(t.n,t.g)
        p=np.clip(-np.expm1(np.log1p(-initiation).sum(axis=1)),1e-10,1)
    else:
        p=predict_probability(x,y,m[eligible],t.c,t.x)
    raw=np.vstack((W0[:,-1],1/p))
    if t.label=="S6":
        pid,hour,z,event,_=t.xlib
        discharge=predict_probability(z,event,m[pid],t.c)
        extra=np.zeros((t.n,t.g))
        extra[pid,hour]=-np.log1p(-discharge)
        exit_w=np.exp(np.cumsum(extra,axis=1))
        early[1,:,1:]=exit_w[:,:-1]
        raw[1]*=exit_w[:,-1]
    end=raw.copy()
    if t.rule is not False:
        for a in (0,1):
            incl=t.landmark[a]
            limits=weighted_quantile(raw[a,incl],m[incl],[t.rule,1-t.rule])
            end[a]=np.clip(raw[a],*limits)
    bounded=0
    if t.tilt is not None:
        target=np.average(t.tilt,weights=m)
        for a in (0,1):
            ix=t.landmark[a]
            share=np.average(t.tilt[ix],weights=m[ix]*end[a,ix])
            require(0<share<1 and 0<target<1,"Bootstrap calibration lost one binary level")
            mult=np.array([(1-target)/(1-share),target/share])
            bounded+=int(np.any((mult<.7)|(mult>1.4)))
            end[a]*=np.clip(mult,.7,1.4)[t.tilt.astype(int)]
    require(np.isfinite(end).all() and np.isfinite(early).all(),"Nonfinite time-updated IPCW")
    return Weights(early,end,bounded,raw)

def frozen_weights(frame,t):
    parts=[frame.loc[frame.arm.eq(a)].sort_values("cohort_id") for a in (0,1)]
    require(all(len(x)==t.n for x in parts),"Frozen weight key count mismatch")
    return Weights(np.stack([d[[f"w_start_{h}" for h in range(t.g)]].to_numpy() for d in parts]),
                   np.stack([d.w_after_g.to_numpy() for d in parts]))

def time_bin(x,limit):
    return np.clip(np.ceil(x-1e-9),1,limit).astype(int)-1

def div(num,den):
    return np.divide(num,den,out=np.zeros_like(num,dtype=float),where=den>0)

def rates(t,w,m,a,when,event,horizon,competing=None):
    when=np.maximum(0,np.asarray(when,dtype=float))
    event=np.asarray(event,bool)&(when<=horizon+1e-9)
    art=t.art[a]
    if competing is None:
        stop=np.minimum(np.minimum(when,art),horizon)
        other=np.zeros(t.n,dtype=bool)
        other_time=when
    else:
        other_time,other=competing
        other=np.asarray(other,bool)&(other_time<=horizon+1e-9)&(other_time<=when+1e-9)
        stop=np.minimum.reduce([when,art,np.where(other,other_time,horizon),np.full(t.n,horizon)])
    died=event & (when<=stop+1e-9)
    other=other & (other_time<=stop+1e-9) & ~died
    end_bin=time_bin(stop,horizon)
    event_bin=time_bin(when,horizon)
    ce_bin=time_bin(other_time,horizon)
    raw_risk=np.cumsum(np.bincount(end_bin,weights=m,minlength=horizon)[::-1])[::-1]
    risk=np.cumsum(np.bincount(end_bin,weights=m*w.late[a],minlength=horizon)[::-1])[::-1]
    de=np.bincount(event_bin[died],weights=(m*w.late[a])[died],minlength=horizon)
    ce=np.bincount(ce_bin[other],weights=(m*w.late[a])[other],minlength=horizon)
    for k in range(min(horizon,t.g)):
        weights=m*w.early[a,:,k]
        risk[k]=weights[end_bin>=k].sum()
        de[k]=weights[died&(event_bin==k)].sum()
        ce[k]=weights[other&(ce_bin==k)].sum()
    hd,hc=div(de,risk),div(ce,risk)
    require(np.all(hd+hc<=1.0000001),"Event weight exceeds at-risk weight")
    alive=np.cumprod(np.clip(1-hd-hc,0,1))
    death=np.cumsum(np.r_[1.,alive[:-1]]*hd) if competing is not None else 1-alive
    return death,raw_risk,int(np.dot(died,m))

def primary_def(t):
    if t.db=="mimic":
        return t.primary_time,t.primary_event
    return np.where(t.primary_event,t.primary_time,672.),t.primary_event

def curve_summary(curve):
    p=curve[:,-1]
    rr=p[1]/p[0] if p[0]>0 else np.nan
    return np.array([p[0],p[1],100*(p[1]-p[0]),rr,np.sum(curve[0]-curve[1])/24])

def primary(t,w,m):
    when,evt=primary_def(t)
    return [rates(t,w,m,a,when,evt,672) for a in (0,1)]

def cox(t,w,m):
    observed_time=np.minimum(t.primary_time,t.hosp_rel) if t.db=="eicu" else t.primary_time
    evt=t.primary_event.copy()
    status=evt[None,:] & (observed_time[None,:]<=t.art+1e-9) & (m[None,:]>0)
    times=np.unique(np.tile(observed_time,(2,1))[status])
    if len(times)==0:
        return np.full(3,np.nan)
    bins=np.maximum(np.ceil(times-1e-9).astype(int)-1,0)
    risk=np.empty((2,len(times)))
    d=np.zeros((2,len(times)))
    stop=np.minimum(observed_time[None,:],t.art)
    for a in (0,1):
        order=np.argsort(stop[a],kind="stable")
        event_times=observed_time[status[a]]
        event_idxs=np.searchsorted(times,event_times)
        person=np.flatnonzero(status[a])
        eb=bins[event_idxs]
        ew=np.where(eb<t.g,w.early[a,person,np.minimum(eb,t.g-1)],w.late[a,person])
        np.add.at(d[a],event_idxs,m[person]*ew)
        for k in range(t.g+1):
            at=bins==k if k<t.g else bins>=t.g
            if not at.any():
                continue
            wk=m*(w.early[a,:,k] if k<t.g else w.late[a])
            risk_suffix=np.r_[np.cumsum(wk[order][::-1])[::-1],0.]
            idx=np.searchsorted(stop[a,order],times[at]-1e-9)
            risk[a,at]=risk_suffix[idx]
    results=[]
    for prefix in (np.ones(len(times),dtype=bool),times<=168.,times>168.):
        ix=prefix & (risk[0]>0)&(risk[1]>0)
        nd=d[:,ix].sum(axis=0)
        t1=d[1,ix].sum()
        shift=np.log(risk[1,ix])-np.log(risk[0,ix])
        if not 0<t1<nd.sum():
            results.append(np.nan)
            continue
        score=lambda beta:t1-np.dot(nd,expit(shift+beta))
        results.append(np.exp(brentq(score,-40,40)))
    return np.asarray(results)

def terminal_support(t,w,m):
    when,evt=primary_def(t)
    eligible=(t.art>=t.g)&(m[None,:]>0)
    early_death=evt[None,:]&(when[None,:]<=t.g)&(when[None,:]<=t.art+1e-9)
    after=eligible&np.isinf(t.art)&~early_death
    return early_death,after

def free_days(t,w,m,main):
    _,after=terminal_support(t,w,m)
    means=[]
    for a in (0,1):
        alive_after=1-main[a][0][t.g-1]
        has=after[a]&np.isfinite(t.free)
        weights=w.late[a,has]*m[has]
        means.append(alive_after*np.average(t.free[has],weights=weights) if weights.sum()>0 else np.nan)
    return np.array([*means,means[1]-means[0]])

def hb_curve(t,w,m,main):
    early_death,after=terminal_support(t,w,m)
    _,event=primary_def(t)
    risks=[]
    for a in (0,1):
        ed=early_death[a]
        aft=after[a]
        use=ed|aft
        weights=np.zeros(t.n)
        ew=w.early[a,np.flatnonzero(ed),time_bin(t.primary_time[ed],672).clip(max=t.g-1)]*m[ed]
        mass=main[a][0][t.g-1]
        if ew.sum()>0:
            weights[ed]=mass*ew/ew.sum()
        post=w.late[a,aft]*m[aft]
        if post.sum()>0:
            weights[aft]=(1-mass)*post/post.sum()
        y=event.astype(int)
        if len(np.unique(y[use&(weights>0)]))<2:
            risks.append(np.full(20,np.nan))
        else:
            risks.append(predict_probability(t.hb_fit,y,weights*m.sum(),10.0,t.hb_predict))
    r0,r1=risks
    return np.stack((r0,r1,100*(r1-r0),np.divide(r1,r0)))

def subgroup(t,w,m):
    pairs={}
    for name,partition in t.group.items():
        flags=partition if isinstance(partition,tuple) else (partition,~partition)
        pair=[]
        for selected in flags:
            sub=m*selected
            if sub.sum()==0:
                pair.append(np.full(5,np.nan))
            else:
                # Within subgroup, each arm's risk-set weights can be multiplied by
                # its marginal stabilization factor at each step without changing h(t).
                xs=primary(t,w,sub)
                pair.append(curve_summary(np.stack([z[0] for z in xs])))
        a,b=pair
        rr=np.log(a[3]/b[3]) if a[3]>0 and b[3]>0 else np.nan
        pairs[name]=np.stack((a,b,[np.nan,np.nan,a[2]-b[2],rr,a[4]-b[4]]))
    return pairs

def estimate(t,m, w=None, extended=False):
    w=estimate_weights(t,m) if w is None else w
    cur=primary(t,w,m)
    curve=np.stack([r[0] for r in cur])
    out={"metrics":curve_summary(curve),"events":np.array([r[2] for r in cur]),"tilt_bounded":w.tilt_bounded}
    if not extended:
        return out
    out.update({"curve":curve,"risk":np.stack([r[1] for r in cur]),"cox":cox(t,w,m),
                "free":free_days(t,w,m,cur),"hb":hb_curve(t,w,m,cur),"groups":subgroup(t,w,m)})
    endpoints=t.secondary.copy()
    if t.db=="eicu":
        endpoints["28-day in-hospital death (Aalen–Johansen)"]=(np.minimum(t.hosp_rel,672),t.primary_event,672)
    second={}
    for title,(when,event,horizon) in endpoints.items():
        horizon=int(max(1,np.ceil(when.max()))) if horizon is None else horizon
        if "Aalen" in title:
            comp=(t.hosp_rel,~t.hosp_death)
        elif "discharge-censored" in title:
            comp=None
        else:
            comp=(when,~event)
        ss=[rates(t,w,m,a,when,event,horizon,comp) for a in (0,1)]
        metric=curve_summary(np.stack([z[0] for z in ss]))
        second[title]=np.concatenate((metric[:4],np.array([z[2] for z in ss])))
    out["secondary"]=second
    return out
