from settings import *
from weight_model_inputs import BINARY,COHORT_SPECIFIC,CONTINUOUS,IMPUTE,LAB_CATS,MULTI
from dataclasses import dataclass
import re
import warnings
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.exceptions import ConvergenceWarning

SPLINES={"baseline_hb":3,"t0_from_icu_admit_hr":2,"age":2,"sbp_min_6h":2}
LOG_LINEAR=("creatinine","bun_cr_ratio","wbc","platelets")
MISSING_FLAGS=("heme_renal_missing_flag","inr_missing_flag","vitals_missing_flag","inr_pt_proxy_flag")
CUTS={"lactate":(2,4),"albumin":(2.5,3.2),"bilirubin":(1.2,3),"ptt":(35,50)}
LEVELS={
    "lactate":("<2.0","2.0–3.9",">=4.0"), "albumin":("<2.5","2.5–3.1",">=3.2"),
    "bilirubin":("<1.2","1.2–2.9",">=3.0"), "ptt":("<35","35–49",">=50"),
}

@dataclass
class Features:
    values: pd.DataFrame
    category: pd.DataFrame
    model: pd.DataFrame
    checks: dict

def race(val,db):
    label=str(val).strip().lower()
    if re.search(r"unknown|declined|unable|unobtain|not specified|^nan$|^none$|^<na>$",label):
        return "Unknown"
    if "white" in label or "caucasian" in label:
        return "White" if db=="mimic" else "Caucasian"
    if "black" in label or "african" in label:
        return "Black" if db=="mimic" else "African_American"
    if "hispanic" in label or "latin" in label:
        return "Hispanic"
    return "Asian_Other"

def icu(val,db):
    label=str(val).upper()
    if db=="mimic":
        if re.search(r"CCU|CVICU|CARDI|CORONARY",label):
            return "Cardiac_ICU"
        if re.search(r"MICU/SICU|MEDICAL/SURGICAL",label):
            return "MICU_SICU"
        if "MICU" in label or "MEDICAL INTENSIVE" in label:
            return "MICU"
        if re.search(r"SURG|SICU|NEURO|TRAUMA|PACU",label):
            return "Surgical_Trauma_Neuro_ICU"
        raise RuntimeError("Unmapped MIMIC ICU category")
    if re.search(r"CCU|CTICU|CVICU|CSICU|CARDI|CORONARY",label):
        return "Cardiac_ICU"
    if db=="eicu" and re.search(r"MED.?SURG|MEDICAL/SURGICAL|MED/SURG",label):
        return "Med_Surg_ICU"
    if re.search(r"MICU|MEDICAL INTENSIVE",label):
        return "MICU"
    if re.search(r"SICU|SURGICAL INTENSIVE",label):
        return "SICU_TSICU" if db=="mimic" else "SICU"
    return "Other_ICU" if db=="mimic" else "Neuro_Other_ICU"

def standardized(x):
    scale=x.std(ddof=0)
    return (x-x.mean())/scale if scale>0 else x*0

def ns(x,df):
    basis=dmatrix(f"cr(x, df={df}, constraints='center') - 1",{"x":x},return_type="dataframe")
    return basis.set_axis([f"{x.name}:ns{index+1}" for index in range(df)],axis=1).set_axis(x.index)

def design_matrix(values,category,extra_spline=()):
    terms={}
    for name,x in values.items():
        if name in SPLINES:
            z=np.log1p(x) if name=="t0_from_icu_admit_hr" else x
            terms.update({k:standardized(v) for k,v in ns(z,SPLINES[name]+int(name in extra_spline)).items()})
        elif name in extra_spline:
            terms.update({k:standardized(v) for k,v in ns(x,2).items()})
        else:
            terms[name]=standardized(np.log1p(x) if name in LOG_LINEAR else x)
    dummies=pd.get_dummies(category,prefix_sep=" = ",drop_first=True,dtype=float)
    return pd.concat([pd.DataFrame(terms,index=values.index),dummies],axis=1)

def encode_baseline(b,db):
    val=b[list(CONTINUOUS)].astype(float).copy()
    cat=b[list(MULTI)].astype("string").fillna("Unknown")
    cat["race_eth"]=cat.race_eth.map(lambda x:race(x,db))
    cat["icu_unit_type"]=cat.icu_unit_type.map(lambda x:icu(x,db))
    observed=b.observed_strategy.isin(["early","restrictive"])
    source_counts=pd.crosstab(cat.loc[observed,"admit_source"],b.loc[observed,"observed_strategy"])
    combine=source_counts.index[source_counts.reindex(columns=["early","restrictive"],fill_value=0).min(axis=1)<15]
    cat["admit_source"]=cat.admit_source.replace(dict.fromkeys(combine,"Ward_Other"))
    for name in (*BINARY,*COHORT_SPECIFIC[db]):
        flag=b[name].astype("boolean")
        cat[name]=flag.astype("string").fillna("Unknown")
    vitals=("hr_last","hr_max_6h","sbp_last","sbp_min_6h","map_last","map_min_6h","rr_last","spo2_min_6h","shock_index_t0")
    renal=("platelets","creatinine","bun","bun_cr_ratio","wbc","bicarbonate")
    cat["vitals_missing_flag"]=val[list(vitals)].isna().any(axis=1).map({True:"true",False:"false"})
    cat["heme_renal_missing_flag"]=val[list(renal)].isna().any(axis=1).map({True:"true",False:"false"})
    cat["inr_missing_flag"]=val.inr.isna().map({True:"true",False:"false"})
    calibrated=val.inr.isna() & b.pt.notna() if db=="eicu" else pd.Series(False,index=b.index)
    if calibrated.any():
        paired=b.pt.notna() & val.inr.notna()
        require(paired.sum()>=30,"Too few paired PT/INR values for eICU calibration")
        regression=LinearRegression().fit(b.loc[paired,["pt"]],val.loc[paired,"inr"])
        val.loc[calibrated,"inr"]=np.clip(regression.predict(b.loc[calibrated,["pt"]]),0.4,25)
    cat["inr_pt_proxy_flag"]=calibrated.map({True:"true",False:"false"})
    val["hb_prior_drop_rate"]=val.hb_prior_drop_rate.fillna(0)
    for name in LAB_CATS:
        cuts=CUTS[name]
        cat[f"{name}_cat"]=pd.cut(b[name],[-np.inf,*cuts,np.inf],right=False,labels=LEVELS[name]).astype("string").fillna("Unmeasured")
    require(val[["baseline_hb","age","t0_from_icu_admit_hr"]].notna().all().all(),"cohort and cloning time-zero baseline drift")
    check_input=pd.concat([val,pd.get_dummies(cat,drop_first=True,dtype=float)],axis=1)
    require(not check_input.columns.duplicated().any(),"Duplicated imputation feature")
    lo=check_input.min().fillna(-np.inf)
    hi=check_input.max().fillna(np.inf)
    lo=lo.where(hi>lo,-np.inf)
    hi=hi.where(hi>lo,np.inf)
    imputer=IterativeImputer(estimator=BayesianRidge(),max_iter=100,tol=0.001,random_state=SEED,
                            skip_complete=True,min_value=lo.to_numpy(),max_value=hi.to_numpy(),keep_empty_features=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        filled=imputer.fit_transform(check_input)
    val=val.assign(**{name:filled[:,check_input.columns.get_loc(name)] for name in IMPUTE})
    require(np.isfinite(val.to_numpy()).all(),"Imputation returned missing or nonfinite baseline covariates")
    diag={"PT_calibrated":masked(int(calibrated.sum())),"imputer_iterations":imputer.n_iter_,
          "imputer_converged":not any(isinstance(w.message,ConvergenceWarning) for w in caught),
          "merged_admit_source_levels":len(combine)}
    require(diag["imputer_converged"],"BayesianRidge iterative imputation did not converge with the specified tolerance")
    model=design_matrix(val,cat)
    require(np.isfinite(model.to_numpy()).all(),"Nonfinite model design")
    return Features(val,cat,model,diag)
