from settings import *
from model_features import design_matrix, SPLINES, LOG_LINEAR
from ipcw import fit_ipcw
from weight_model_inputs import load_blinded
from model_features import encode_baseline
import math
from itertools import product
import numpy as np
import pandas as pd

def moments(x,w):
    avg=np.average(x,weights=w)
    var=np.average((x-avg)**2,weights=w)
    return float(avg),float(var)

def smd(mu0,var0,mu1,var1):
    pooled=math.sqrt((var0+var1)/2)
    if pooled>0:
        return abs(mu1-mu0)/pooled
    return 0.0 if mu0==mu1 else None

def weights_summary(db,arm,df):
    d=[]
    for col in ("raw","raw_p01_p99","stabilized","stabilized_p01_p99"):
        w=df[col].to_numpy()
        require(np.isfinite(w).all() and (w>0).all(),"Nonpositive or nonfinite weight")
        ess=w.sum()**2/np.dot(w,w)
        require(0<ess<=len(w)+1e-7,"Kish ESS invariant failed")
        q01,q25,q50,q75,q99=np.quantile(w,[0.01,0.25,0.50,0.75,0.99])
        d.append({"database":db,"arm":arm,"weight":col,"n":len(w),"mean":w.mean(),"sd":w.std(ddof=1),"median":q50,
                  "q1":q25,"q3":q75,"iqr":q75-q25,"p01":q01,"p99":q99,"min":w.min(),"max":w.max(),
                  "ess":ess,"ess_fraction":ess/len(w)})
    return d

def diagnose(db,b,f,fit,design):
    # Always evaluate the four prespecified raw continuous measurements, all original
    # baseline covariates, and the nonlinear terms used in the denominator models.
    model_nonlinear=design[[x for x in design.columns if ":ns" in x or ":Hb" in x]]
    onehot=pd.get_dummies(f.category,prefix_sep=" = ",drop_first=False,dtype=float)
    transformed=np.log1p(f.values[list(LOG_LINEAR)]).rename(columns=lambda c:f"log1p({c})")
    mat=pd.concat([f.values,transformed,model_nonlinear,onehot],axis=1).set_axis(b.cohort_id,axis=0)
    observed=b.set_index("cohort_id").observed_strategy
    ids0,ids1=fit.controls.cohort_id,fit.intervention.cohort_id
    w0,w1=(x.raw_p01_p99.to_numpy() for x in (fit.controls,fit.intervention))
    sw0,sw1=(x.stabilized_p01_p99.to_numpy() for x in (fit.controls,fit.intervention))
    rows=[]
    for name,col in mat.items():
        x0=col.loc[ids0].to_numpy()
        x1=col.loc[ids1].to_numpy()
        pre0=col.loc[observed.eq("restrictive")].to_numpy()
        pre1=col.loc[observed.eq("early")].to_numpy()
        m0,v0=moments(x0,w0)
        m1,v1=moments(x1,w1)
        s=smd(m0,v0,m1,v1)
        structural="balanced_constant" if v0+v1==0 and m0==m1 else "evaluated"
        if s is None:
            structural="no_variance_between_different_values"
        rows.append({
            "database":db,"feature":name,"type":"indicator" if name in onehot else "continuous",
            "prespecified_gate_term":":Hb" not in name,
            "observed_unweighted_smd":smd(pre0.mean(),pre0.var(),pre1.mean(),pre1.var()),
            "unweighted_smd":smd(x0.mean(),x0.var(),x1.mean(),x1.var()),"weighted_smd":s,
            "stabilized_weighted_smd":smd(*moments(x0,sw0),*moments(x1,sw1)),
            "weighted_arm0_mean":m0,"weighted_arm1_mean":m1,"unweighted_arm0_mean":x0.mean(),"unweighted_arm1_mean":x1.mean(),
            "unweighted_arm0_count":masked(int(x0.sum())) if name in onehot else "—",
            "unweighted_arm1_count":masked(int(x1.sum())) if name in onehot else "—",
            "weighted_variance_ratio":v1/v0 if v0>0 and v1>0 else "Not estimable",
            "balance_status":structural if s is not None and s<SMD_LIMIT else "failed",
        })
    table=pd.DataFrame(rows)
    weights=pd.DataFrame([*weights_summary(db,0,fit.controls),*weights_summary(db,1,fit.intervention)])
    core=table.loc[table.prespecified_gate_term]
    stab=weights.loc[weights.weight.eq("stabilized_p01_p99")]
    raw=weights.loc[weights.weight.eq("raw_p01_p99")]
    mean_ok=bool(stab["mean"].between(*MEAN_RANGE).all())
    ess_ok=bool((raw.ess.ge(ESS_ABS)&raw.ess_fraction.ge(ESS_FRACTION)).all())
    result={
        "max_smd":float(core.weighted_smd.max()),
        "max_smd_feature":str(core.loc[core.weighted_smd.idxmax(),"feature"]),
        "failed_smd_count":int(core.balance_status.eq("failed").sum()),
        "max_stabilized_smd":float(core.stabilized_weighted_smd.max()),
        "weight_mean_ok":mean_ok,"ess_ok":ess_ok,
        "passes":bool(core.balance_status.ne("failed").all() and mean_ok and ess_ok),**fit.fit_counts,
        **f.checks,
    }
    return table,weights,result
