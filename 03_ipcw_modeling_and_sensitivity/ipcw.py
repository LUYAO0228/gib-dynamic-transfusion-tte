from settings import *
from weight_model_inputs import TIME_VARYING
from model_features import standardized
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

@dataclass
class FittedWeights:
    controls: pd.DataFrame
    intervention: pd.DataFrame
    fit_counts: dict
    bounds: dict
    hourly: pd.DataFrame
    full_liberal: pd.DataFrame

def probability(x,y,predict=None,c_penalty=2.0,sample_weight=None):
    require(set(y)=={0,1},"Censoring model requires both classes")
    model=LogisticRegression(C=c_penalty,solver="lbfgs",max_iter=4000,tol=1e-9,random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        model.fit(x,y,sample_weight=sample_weight)
    probs=model.predict_proba(x if predict is None else predict)[:,1]
    require(np.isfinite(probs).all() and ((probs>0)&(probs<1)).all(),"Invalid estimated IPCW probability")
    return probs

def finalize(tab,arm,eligible):
    retained=tab.loc[eligible].copy()
    require(len(retained)>0 and retained.cohort_id.is_unique,f"Empty or duplicate arm {arm} balance cohort")
    bounds={}
    for label in ("raw","stabilized"):
        low,high=retained[label].quantile([0.01,0.99])
        bounds[label]=(float(low),float(high))
        retained[f"{label}_p01_p99"]=retained[label].clip(low,high)
    return retained,bounds

def fit_ipcw(b,h,p,f,design=None,c_penalty=2.0,late_antithrom=False):
    model=(f.model if design is None else design).set_axis(b.cohort_id,axis=0)
    arm0=h.loc[h.arm.eq(0)&h.at_risk_for_censor.eq(1)].sort_values(["cohort_id","hour_idx"]).copy()
    require(arm0.groupby("cohort_id").size().size==len(b),"Arm 0 has no initial risk row for some participants")
    a0=p.loc[p.arm.eq(0)].set_index("cohort_id")
    require(arm0.groupby("cohort_id").early_prbc_censor_event.sum().sort_index().equals(
        a0.artificial_censored.sort_index().astype(arm0.early_prbc_censor_event.dtype)),
        "Arm 0 censor event totals disagree between blinded inputs")
    hourly=arm0[list(TIME_VARYING)].astype(float)
    missing=hourly.sbp_locf.isna()
    sbp=f.values.sbp_last.set_axis(b.cohort_id)
    hourly.loc[missing,"sbp_locf"]=arm0.loc[missing,"cohort_id"].map(sbp)
    require(hourly.notna().all().all(),"Nonfinite time-varying design")
    for name in ("hb_change_from_t0","sbp_locf"):
        hourly[name]=standardized(hourly[name])
    # The missingness flag is already represented at baseline, and the current-hour SBP
    # retains its strictly observed fallback to the imputed baseline SBP.
    steps=pd.get_dummies(arm0.hour_idx,prefix="hour",drop_first=True,dtype=float)
    x0=np.column_stack((model.loc[arm0.cohort_id],hourly,steps))
    if late_antithrom:
        flag=f.category.antiplatelet_anticoag_pre_t0.eq("True").set_axis(b.cohort_id)
        x0=np.column_stack((x0,flag.loc[arm0.cohort_id].to_numpy()*arm0.hour_idx.ge(3).to_numpy()))
    y0=arm0.early_prbc_censor_event.to_numpy(dtype=int)
    den0=probability(x0,y0,c_penalty=c_penalty)
    num0=arm0.groupby("hour_idx").early_prbc_censor_event.transform("mean").to_numpy()
    arm0["log_inverse"]=-np.log1p(-den0)
    arm0["log_stabilized"]=np.log1p(-num0)-np.log1p(-den0)
    arm0["raw"]=np.exp(arm0.groupby("cohort_id").log_inverse.cumsum())
    arm0["stabilized"]=np.exp(arm0.groupby("cohort_id").log_stabilized.cumsum())
    last=arm0.groupby("cohort_id",sort=False).tail(1).set_index("cohort_id",drop=False)
    eligible0=a0.eligible_g_cross_sectional_flag.astype(bool) & a0.artificial_censored.eq(0)
    ret0,b0=finalize(last,0,eligible0.reindex(last.index))

    liberal=p.loc[p.arm.eq(1)].sort_values("cohort_id").copy()
    liberal["in_fit"]=liberal.liberal_completed_by_g | liberal.artificial_censored_if_exit_at_g
    require(not (liberal.liberal_completed_by_g & liberal.artificial_censored_if_exit_at_g).any(),
            "Incompatible simultaneous liberal completion and censoring")
    fit=liberal.loc[liberal.in_fit].copy()
    y1=fit.liberal_completed_by_g.to_numpy(dtype=int)
    fit["p_complete"]=probability(model.loc[fit.cohort_id].to_numpy(),y1,c_penalty=c_penalty)
    fit["raw"]=1/fit.p_complete
    fit["stabilized"]=float(y1.mean())/fit.p_complete
    eligible1=fit.liberal_completed_by_g & fit.eligible_g_cross_sectional_flag
    ret1,b1=finalize(fit,1,eligible1)

    counts={"restrictive_risk_rows":len(arm0), "liberal_fit_kept":int(y1.sum()),
        "liberal_fit_censored":int(len(y1)-y1.sum()),"arm0_balance":len(ret0),"arm1_balance":len(ret1)}
    return FittedWeights(ret0,ret1,counts,{0:b0,1:b1},arm0,fit)
