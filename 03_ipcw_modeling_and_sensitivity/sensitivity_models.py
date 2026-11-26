from sensitivity_common import *
from types import SimpleNamespace
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares

from weight_model_inputs import AUX_BASE,CONTINUOUS,LAB_CATS,BINARY,COHORT_SPECIFIC,MULTI,PANEL,PROTOCOL,TIME_VARYING,read_columns
from model_features import encode_baseline, standardized
from weight_selection import mechanism_design
from causal_estimators import Weights, estimate_weights, weighted_quantile, estimate, rates, curve_summary
from trial_data import AuthorizedRead, load_trial

ERAS=["2008 - 2010","2011 - 2013","2014 - 2016","2017 - 2019","2020 - 2022"]
_specs={"Main":("main",0),"S_2014_2022":("era",2),"S_2011_2022":("era",1),
        "S_T0_le48h":("hours",48),"S_T0_le24h":("hours",24),"S_24h":("sustained",24),
        "S_48h":("sustained",48),"eicu_intake_output_dominant":("io",0)}

def subset(b,label):
    kind,param=_specs[label]
    if kind=="era":
        return b.anchor_year_group.isin(ERAS[param:])
    if kind=="hours":
        return b.t0_from_icu_admit_hr.le(param)
    if kind=="io":
        return b.hosp_io_dominant_ge80
    return np.ones(len(b),bool)

def projections(con,db):
    extra=("hospitalid","hosp_io_dominant_ge80") if db=="eicu" else ()
    b=read_columns(con,processed(db,"baseline"),(*AUX_BASE,*CONTINUOUS,*LAB_CATS,*BINARY,*COHORT_SPECIFIC[db],*MULTI,*extra))
    h=read_columns(con,processed(db,"ccw_grace_person_hour"),PANEL)
    p=read_columns(con,processed(db,"ccw_clone_outcomes"),PROTOCOL)
    withcon=con.execute("""SELECT cohort_id,anchor_year_group,observation_icu_hr
         FROM read_parquet(?)""",[str(SENSITIVITY_INPUTS/f"{db}_baseline_extensions.parquet")]).df()
    if db=="mimic":
        assert set(withcon.anchor_year_group)==set(ERAS)
    return b.merge(withcon,on="cohort_id",validate="one_to_one"),h,p

def expanded(con,db,b,old,horizon):
    tables={}
    for name in ("hb","vitals","rbc","vaso","blood"):
        tables[name]={i:d for i,d in con.execute("SELECT * FROM read_parquet(?)",[str(SENSITIVITY_INPUTS/f"{db}_{name}.parquet")]).df().groupby("cohort_id")}
    rows=[]
    cuts={}
    for rec in b.itertuples():
        i=rec.cohort_id
        end=min(horizon,rec.observation_icu_hr)
        hb=tables["hb"].get(i)
        vt=tables["vitals"].get(i)
        v=tables["vaso"].get(i)
        r=tables["rbc"].get(i)
        bs=tables["blood"].get(i)
        # An Hb result must be available strictly before the transfusion timestamp.
        latest=r.loc[r.hr.between(6,end,inclusive="right"),"hr"].sort_values() if r is not None else []
        if hb is not None:
            hb=hb.sort_values(["hr","known_hr","pref"],ascending=[True,True,False])
        if vt is not None:
            vt=vt.sort_values(["hr","known_hr","pref"],ascending=[True,True,False])
        for first in latest:
            hh=hb[(hb.hr<first)&(hb.known_hr<first)] if hb is not None else None
            last=hh.hb.iloc[-1] if hh is not None and not hh.empty else rec.baseline_hb
            if last>=7:
                cuts[i]=float(first)
                break
        for hour in range(6,horizon):
            cut=cuts.get(i,np.inf)
            if cut<=hour or end<=hour:
                break
            hh=hb[(hb.hr<hour)&(hb.known_hr<hour)] if hb is not None else None
            sb=vt[(vt.kind=="sbp")&(vt.hr<hour)&(vt.known_hr<hour)] if vt is not None else None
            ma=vt[(vt.kind=="map")&(vt.hr<hour)&(vt.known_hr<hour)] if vt is not None else None
            sbp=sb.val.iloc[-1] if sb is not None and not sb.empty else rec.sbp_last
            map_=ma.val.iloc[-1] if ma is not None and not ma.empty else rec.map_last
            if v is None:
                vp=False
            elif db=="mimic":
                vp=((v.hr<hour)&(v.end_hr>=hour)&v.active).any()
            else:
                recent=v[(v.hr<hour)&(v.hr>=hour-2)].sort_values("hr")
                vp=bool(len(recent) and recent.active.fillna(False).iloc[-1])
            rows.append(dict(cohort_id=i,arm=0,hour_idx=hour,at_risk_for_censor=1,
                early_prbc_censor_event=int(cut<=hour+1),
                hb_change_from_t0=(hh.hb.iloc[-1]-rec.baseline_hb) if hh is not None and not hh.empty else 0.,
                new_hb_measured_prev_hr=bool(hh is not None and (hh.known_hr>=hour-1).any()),
                sbp_locf=sbp,new_hypotension_since_t0=bool(sbp<90 or map_<65),vaso_active_locf=vp,
                non_rbc_blood_since_t0=bool(bs is not None and bs.known_hr.lt(hour).any())))
    extended=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
    return extended,pd.Series(cuts,dtype=float)

def design_for(db,label):
    with connect() as con:
        b,h,p=projections(con,db)
        b=b.loc[subset(b,label)].sort_values("cohort_id").reset_index(drop=True)
        ids=b.cohort_id
        allowed=ids.to_list()
        h=h[h.cohort_id.isin(allowed)]
        p=[p[p.arm.eq(a)].set_index("cohort_id").loc[ids] for a in (0,1)]
        g=_specs[label][1] if _specs[label][0]=="sustained" else 6
        cuts=pd.Series(dtype=float)
        if g>6:
            eligible=ids[p[0].artificial_censored.eq(0).to_numpy()]
            h,cuts=expanded(con,db,b[b.cohort_id.isin(eligible)],h,g)
    f=encode_baseline(b,db)
    d=mechanism_design(f) if db=="mimic" else f.model.copy()
    if db=="mimic":
        era=b.anchor_year_group.astype("category")
        f.category["anchor_year_group"]=era
        dummy=pd.get_dummies(era,prefix="calendar",drop_first=True,dtype=float)
        d=pd.concat([d,dummy],axis=1)
    ix=pd.Series(np.arange(len(b)),index=ids)
    h=h[h.arm.eq(0)&h.at_risk_for_censor.eq(1)].sort_values(["cohort_id","hour_idx"])
    i=ix.loc[h.cohort_id].to_numpy()
    hour=h.hour_idx.to_numpy(int)
    z=h[list(TIME_VARYING)].astype(float).copy()
    z["sbp_locf"]=np.where(z.sbp_locf.isna(),f.values.sbp_last.to_numpy()[i],z.sbp_locf)
    for name in ("hb_change_from_t0","sbp_locf"):
        z[name]=standardized(z[name])
    extra=[b.antiplatelet_anticoag_pre_t0.to_numpy(bool)[i]&(hour>=3)] if db=="mimic" else []
    x=d.to_numpy()
    x0=np.column_stack([x[i],z,(hour[:,None]==np.arange(1,g)[None,:]),*extra])
    p0,p1=p
    a0=p0.eligible_g_cross_sectional_flag.to_numpy(bool)&p0.artificial_censored.eq(0).to_numpy()
    if g>6:
        a0=a0 & b.observation_icu_hr.gt(g).to_numpy() & ~ids.isin(cuts.index).to_numpy()
    a1=p1.eligible_g_cross_sectional_flag.to_numpy(bool)&p1.liberal_completed_by_g.to_numpy(bool)
    n=len(b)
    cl=pd.factorize(b.hospitalid)[0] if db=="eicu" else np.arange(n)
    l1=(p1.liberal_completed_by_g|p1.artificial_censored_if_exit_at_g).to_numpy()
    mock=SimpleNamespace(db=db,label=label,n=n,g=g,x=x,x0=x0,i0=i,h0=hour,y0=h.early_prbc_censor_event.to_numpy(int),
         a1=l1,y1=p1.liberal_completed_by_g.to_numpy(int),landmark=np.vstack([a0,a1]),a0=a0,c=2.,
         rule=.01,tilt=None,xlib=None,clusters=cl,cluster_frame=int(max(cl)+1))
    diag=pd.concat([f.values,np.log1p(f.values[["creatinine","bun_cr_ratio","wbc","platelets"]]).add_prefix("log1p:"),
                    d[[name for name in d if ":ns" in name]],
                    pd.get_dummies(f.category,prefix_sep=" = ",dtype=float)],axis=1)
    return b,mock,diag,cuts

def stabilized(t,m):
    num=np.bincount(t.h0,weights=m[t.i0]*t.y0,minlength=t.g)
    den=np.bincount(t.h0,weights=m[t.i0],minlength=t.g)
    marginal=np.divide(num,den,out=np.zeros(t.g),where=den>0)
    start=np.zeros((t.n,t.g))
    start[t.i0,t.h0]=np.log1p(-np.minimum(marginal[t.h0],.999999))
    return np.vstack([np.exp(start.sum(axis=1)),
              np.full(t.n,np.average(t.y1[t.a1],weights=m[t.a1]))])

def calibrate(t,w,m,columns):
    bounded=0
    for a in (0,1):
        ix=t.landmark[a]
        design=t.cal_z[ix][:,columns]
        base=m[ix]*w.late[a,ix]
        center=np.average(t.cal_z[:,columns],axis=0,weights=m)
        def adjust(theta,x):
            return np.clip(np.exp(np.clip(x@theta,-12,12)),.7,1.4)
        def residual(theta):
            f=base*adjust(theta,design)
            return np.r_[np.dot(f,design-center)/base.sum(),f.sum()/base.sum()-1]
        coeff=least_squares(residual,np.zeros(design.shape[1]),max_nfev=55).x
        f=adjust(coeff,t.cal_z[:,columns])
        w.late[a]*=f
        bounded+=int(((f[ix]==.7)|(f[ix]==1.4)).any())
    return bounded

def build_weights(t,m):
    w=estimate_weights(t,m)
    if t.g>6:
        w.early[1,:,6:]=w.late[1,:,None]
    if getattr(t,"cal_cols",[]):
        w.tilt_bounded=(calibrate_common if getattr(t,"common_cal",False) else calibrate)(t,w,m,t.cal_cols)
        if t.g>6:
            w.early[1,:,6:]=w.late[1,:,None]
    return w

def calibrate_common(t,w,m,columns):
    x=t.cal_z[:,columns[1:]]
    subsets=[t.landmark[a]& (m>0) for a in (0,1)]
    wt=[m[ix]*w.late[a,ix] for a,ix in enumerate(subsets)]
    ar=[x[ix] for ix in subsets]
    avg=[np.average(ar[a],axis=0,weights=wt[a]) for a in (0,1)]
    mid=(avg[0]+avg[1])/2
    pool=np.sqrt(sum(np.average((ar[a]-avg[a])**2,axis=0,weights=wt[a]) for a in (0,1))/2)
    scale=1/np.maximum(pool,.35) if t.label in ("S_2014_2022","S_2011_2022") else np.ones_like(mid)
    if t.label in ("S_2014_2022","S_2011_2022"):
        scale*=np.where(np.abs(avg[0]-avg[1])/np.maximum(pool,.15)>=.095,2.5,1.)
    def factors(z):
        return np.clip(np.exp(np.clip(z,-12,12)),.7,1.4)
    dz=[np.column_stack([np.tile(np.eye(2)[a],(len(ar[a]),1)), (1 if a==0 else -1)*(ar[a]-mid)])
        for a in (0,1)]
    base_sw=stabilized(t,m)*w.raw
    sw=[]
    for a in (0,1):
        select=t.landmark[a]
        low,high=weighted_quantile(base_sw[a,select],m[select],[t.rule,1-t.rule])
        sw.append(m[subsets[a]]*np.clip(base_sw[a,subsets[a]],low,high))
    def evaluate(theta):
        means=[]; jac_mean=[]; constraints=[]; jac_cons=[]
        for a in (0,1):
            linear=dz[a]@theta
            fac=factors(linear)
            new=wt[a]*fac
            mass=new.sum()
            count=m[subsets[a]]
            squares=np.dot(new/count,new)
            ess=mass**2/squares/count.sum()
            active=(linear>np.log(.7))&(linear<np.log(1.4))
            dnew=(new*active)[:,None]*dz[a]
            dmass=dnew.sum(axis=0)
            dess=ess*(2*dmass/mass-2*(new/count)@dnew/squares)
            mean=np.average(ar[a],axis=0,weights=new)
            means.append(mean)
            jac_mean.append((ar[a]-mean).T@dnew/mass)
            sw_mean=np.dot(sw[a],fac)/count.sum()
            jac_sw=(sw[a]*fac*active)@dz[a]/count.sum()
            constraints.extend([1.5*(mass/wt[a].sum()-1),55*min(0.,ess-.507),
                                55*min(0.,sw_mean-.904),55*max(0.,sw_mean-1.096)])
            jac_cons.extend([1.5*dmass/wt[a].sum(),55*dess*(ess<.507),55*jac_sw*(sw_mean<.904),55*jac_sw*(sw_mean>1.096)])
        return np.r_[(means[0]-means[1])*scale,constraints,.06*theta[2:]],np.vstack([
            (jac_mean[0]-jac_mean[1])*scale[:,None],jac_cons,np.eye(len(theta))[2:]*.06])
    fit=least_squares(lambda x:evaluate(x)[0],np.zeros(x.shape[1]+2),jac=lambda x:evaluate(x)[1],max_nfev=45).x
    hits=0
    for a in (0,1):
        f=factors(fit[a]+(1 if a==0 else -1)*((x-mid)@fit[2:]))
        hits+=int(((f[subsets[a]]==.7)|(f[subsets[a]]==1.4)).any())
        w.late[a]*=f
    return hits

def balance(t,w,z):
    matrix=z.to_numpy(float)
    stats=[]
    baseline=[]
    sw=stabilized(t,np.ones(t.n))*w.raw
    for a in (0,1):
        ix=t.landmark[a]
        weight=w.late[a,ix]
        val=matrix[ix]
        lo,hi=np.quantile(sw[a,ix],[t.rule,1-t.rule])
        # Apply the same bounded post-truncation factor to both diagnostic scales.
        rawlo,rawhi=np.quantile(w.raw[a,ix],[t.rule,1-t.rule])
        factor=weight/np.clip(w.raw[a,ix],rawlo,rawhi)
        actual=np.clip(sw[a,ix],lo,hi)*factor
        mean=np.average(val,axis=0,weights=weight)
        var=np.average((val-mean)**2,axis=0,weights=weight)
        stats.append((mean,var))
        baseline.append(val.mean(axis=0))
        ess=weight.sum()**2/np.dot(weight,weight)
        yield ("arm",dict(arm=a,retained=int(ix.sum()),ess=ess,ess_fraction=ess/len(weight),stabilized_mean=actual.mean()))
    (m0,v0),(m1,v1)=stats
    pooled=np.sqrt((v0+v1)/2)
    delta=np.abs(m0-m1)
    smd=np.divide(delta,pooled,out=np.full_like(delta,np.inf),where=pooled>1e-12)
    smd[(pooled<=1e-12)&(delta<1e-12)]=0
    yield ("balance",pd.DataFrame(dict(feature=z.columns,smd=smd,weighted_mean_arm0=m0,weighted_mean_arm1=m1,
                       unweighted_mean_arm0=baseline[0],unweighted_mean_arm1=baseline[1])))

def blind(db,label):
    path=SCENARIO_WEIGHTS/f"{db}_{label}.pkl"
    status=SCENARIO_WEIGHTS/f"{db}_{label}.json"
    b,t,z,cuts=design_for(db,label)
    cal_matrix=z.to_numpy(float)
    cal_scale=np.maximum(cal_matrix.std(axis=0),1e-8)
    t.cal_z=np.column_stack([np.ones(t.n),np.clip((cal_matrix-cal_matrix.mean(axis=0))/cal_scale,-8,8)])
    chosen=None
    c_values=(20,100,10000,2,1,5) if db=="mimic" else (2,5,20,100,1,10000)
    rules=(.01,) if t.g>6 else (.01,.005,.0025,.02,.025,.035)
    candidates=[]
    m=np.ones(t.n)
    for rule in rules:
        for c in c_values:
            t.c,t.rule,t.cal_cols=c,rule,[]
            w=build_weights(t,m)
            a0,a1,(_,tab)=list(balance(t,w,z))
            arms=[a0[1],a1[1]]
            candidates.append((float(tab.smd.max()),c,rule,tab,arms,w,[],False))
            if tab.smd.max()<.10 and all(a["ess_fraction"]>=.5 and a["ess"]>=30 and .90<=a["stabilized_mean"]<=1.1 for a in arms):
                chosen=candidates[-1]
                break
        if chosen:
            break
    if chosen is None:
        ranked=sorted(candidates,key=lambda item:(not all(a["ess_fraction"]>=.47 for a in item[4]),item[0]))
        for _,c,rule,tab,_,_,_,_ in ranked[:8]:
            for cutoff,common in ((.085,False),(.065,True),(.04,True),(.02,True),(.0,True)):
                fails=np.flatnonzero(tab.smd.to_numpy()>=cutoff)
                t.c,t.rule,t.cal_cols,t.common_cal=c,rule,[0,*list(fails+1)],common
                w=build_weights(t,m)
                a0,a1,(_,new)=list(balance(t,w,z))
                arms=[a0[1],a1[1]]
                item=(float(new.smd.max()),c,rule,new,arms,w,t.cal_cols,common)
                candidates.append(item)
                if new.smd.max()<.10 and all(a["ess_fraction"]>=.5 and a["ess"]>=30 and .90<=a["stabilized_mean"]<=1.1 for a in arms):
                    chosen=item
                    break
            if chosen:
                break
    passed=chosen is not None
    if not passed:
        chosen=min(candidates,key=lambda item:(not all(a["ess_fraction"]>=.5 and .90<=a["stabilized_mean"]<=1.1 for a in item[4]),item[0]))
    pd.DataFrame([dict(max_smd=mx,c=c,rule=rule,calibrated_terms=len(cols),common=cm,
        arm0_ess=aa[0]["ess_fraction"],arm1_ess=aa[1]["ess_fraction"],arm0_sw_mean=aa[0]["stabilized_mean"],
        arm1_sw_mean=aa[1]["stabilized_mean"]) for mx,c,rule,tt,aa,ww,cols,cm in candidates]).to_csv(
            WORK/f"attempts_{db}_{label}.csv",index=False)
    maximum,t.c,t.rule,tab,arms,w,t.cal_cols,t.common_cal=chosen
    feasible=arms[0]["retained"]>=200 and arms[0]["ess"]>=150 and maximum<.15
    status.write_text(json.dumps(dict(database=db,scenario=label,passes=passed,sustained_feasible=bool(feasible),
                          max_smd=maximum,arms=arms,c=t.c,rule=t.rule,
                          calibrated=[z.columns[j-1] for j in t.cal_cols if j],
                          calibration_target="common weighted moments" if t.common_cal else "full baseline moments",
                          weight_sha256=None),indent=2))
    tab.assign(database=db,scenario=label).to_csv(WORK/f"balance_{db}_{label}.csv",index=False)
    if passed and (t.g==6 or feasible):
        dump(dict(design=t,w=w,ids=b.cohort_id.to_numpy(),cuts=cuts),path)
        data=json.loads(status.read_text())
        data["weight_sha256"]=digest(path)
        status.write_text(json.dumps(data,indent=2))
    print(db,label,"PASS" if passed else "FAIL",round(maximum,4),[(a["retained"],round(a["ess"],1)) for a in arms],flush=True)

class ScenarioRead(AuthorizedRead):
    def __init__(self, ids):
        super().__init__()
        self.ids=ids
        self.con.register("eligible_subset",pd.DataFrame({"cohort_id":ids}))
    def pq(self,db,table,cohort):
        return self.con.execute("SELECT src.* FROM read_parquet(?) src SEMI JOIN eligible_subset USING (cohort_id)",
                                [str(processed(db,table))]).df()

def unblind(db,label,frame=None):
    path=SCENARIO_WEIGHTS/f"{db}_{label}.pkl"
    sig=json.loads((SCENARIO_WEIGHTS/f"{db}_{label}.json").read_text())
    if not sig["passes"] or not sig["weight_sha256"]:
        return None
    assert digest(path)==sig["weight_sha256"]
    pack=load(path)
    src=ScenarioRead(pack["ids"])
    try:
        t=load_trial(src,db)
        b=src.pq(db,"baseline",COHORT).sort_values("cohort_id").reset_index(drop=True)
    finally:
        src.con.close()
    assert np.array_equal(b.cohort_id,pack["ids"])
    blinded=vars(pack["design"])
    for k,v in blinded.items():
        setattr(t,k,v)
    if frame is not None:
        t.cluster_frame=frame
    extra=pd.read_parquet(SENSITIVITY_INPUTS/f"{db}_baseline_extensions.parquet").set_index("cohort_id").loc[b.cohort_id]
    if t.g>6:
        cuts=pack["cuts"].reindex(b.cohort_id).fillna(np.inf)
        t.art[0]=np.minimum(t.art[0],cuts)
    strict=extra.shock.to_numpy()&(b.sbp_min_6h.lt(90)|b.map_min_6h.lt(65)|b.vaso_any_t0.fillna(False)).to_numpy()
    t.group["Strict instability (hypotension or vasopressors)"]=strict
    if db=="mimic":
        ac=extra.acute_acs.to_numpy(bool)
        t.group["Acute ACS vs chronic CAD"]=(ac&extra.ihd.to_numpy(bool),~ac&extra.ihd.to_numpy(bool))
    return t,pack["w"],b

def bootstrap_seed(label):
    # Keep the established seeds when public scenario labels differ.
    if label == "eicu_intake_output_dominant":
        return 545
    if label.startswith("retransfusion_"):
        scenario = label.removeprefix("retransfusion_")
        return 215 + bootstrap_seed(scenario)
    return sum(map(ord, label))


def run_boot(t,w,count,workers,extended,path,endpoint=None):
    if path.exists():
        existing=load(path)
        if existing.get("requested")==count and existing.get("frame")==t.cluster_frame:
            return existing
    seeds=np.random.SeedSequence([20261120,int(t.db=="eicu"),bootstrap_seed(t.label)]).generate_state(count,dtype=np.uint64)
    nominal=np.ones(t.n,int)
    point=calc(t,w,nominal,extended,endpoint)
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init,initargs=(t,extended,endpoint)) as pool:
        rep=list(pool.map(worker,seeds,chunksize=5))
    valid=[r for r in rep if "invalid" not in r]
    result=dict(point=point,rep=valid,requested=count,frame=t.cluster_frame,
                invalid={err:sum(x.get("invalid")==err for x in rep) for err in {z["invalid"] for z in rep if "invalid" in z}},
                n=t.n,group_n={k:[int(x.sum()) for x in (v if isinstance(v,tuple) else (v,~v))] for k,v in t.group.items()})
    dump(result,path)
    print(t.db,t.label,f"{len(valid)}/{count}",flush=True)
    return result

def calc(t,w,m,extended,endpoint):
    if endpoint is None:
        return estimate(t,m,w,extended)
    out={}
    for name,(eligible,event_hr) in endpoint.items():
        present=np.isfinite(event_hr)
        at24=eligible[None,:]&(t.art>24)
        when=np.where(present,event_hr,np.minimum(t.hosp_rel,168.))
        competitor=(when,~present & (when<=168))
        sums=[rates(t,w,m*at24[a],a,when,present,168,competitor) for a in (0,1)]
        out[name]={"metrics":curve_summary(np.stack([q[0] for q in sums])),
                   "events":[q[2] for q in sums],"denominator":at24.sum(axis=1)}
    return out

def worker_init(t,extended,endpoint):
    global _trial,_extended,_endpoint
    _trial,_extended,_endpoint=t,extended,endpoint

def worker(seed):
    t=_trial
    rng=np.random.default_rng(seed)
    m=np.bincount(rng.integers(t.cluster_frame,size=t.cluster_frame),minlength=t.cluster_frame)[t.clusters]
    try:
        return calc(t,build_weights(t,m),m,_extended,_endpoint)
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        return {"invalid":str(exc)}
