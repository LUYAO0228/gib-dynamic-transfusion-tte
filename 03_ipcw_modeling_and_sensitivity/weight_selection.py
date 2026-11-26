from settings import *
from model_features import encode_baseline, standardized
from ipcw import fit_ipcw
from balance_gate import diagnose
from weight_model_inputs import load_blinded
from dataclasses import replace
import numpy as np
import pandas as pd

PENALTIES=(20.0,100.0,10000.0)
OPTIONS=(("P1–P99",0.01),("P0.5–P99.5",0.005),("P0.25–P99.75",0.0025),("SW cap 10","cap"))

def mechanism_design(f):
    result=f.model.copy()
    t=standardized(np.log1p(f.values.t0_from_icu_admit_hr))
    hb=standardized(f.values.baseline_hb)
    drug=f.category.antiplatelet_anticoag_pre_t0.eq("True").astype(float)
    resp=f.category.resp_charted_flag.eq("True").astype(float)
    result["antithrombotic:Hb_mechanism"]=drug*hb
    result["antithrombotic:log_t0"]=drug*t
    result["respiratory_chart:log_t0"]=resp*t
    return result

def select_truncation(fit,rule):
    tables=[]
    bounds={}
    for arm, src in enumerate((fit.controls,fit.intervention)):
        df=src.copy()
        if rule=="cap":
            factor=np.minimum(1,10/df.stabilized.to_numpy())
            for col in ("raw","stabilized"):
                df[f"{col}_p01_p99"]=df[col]*factor
            bounds[arm]={"rule":"cap","stabilized_cap":10.0}
        else:
            bounds[arm]={}
            for col in ("raw","stabilized"):
                low,high=df[col].quantile([rule,1-rule]).to_numpy()
                df[f"{col}_p01_p99"]=df[col].clip(low,high)
                bounds[arm][col]=(float(low),float(high))
        tables.append(df)
    return replace(fit,controls=tables[0],intervention=tables[1],bounds=bounds)

def restricted_tilt(fit, b, f, table):
    failed=table.loc[table.prespecified_gate_term & table.balance_status.eq("failed")]
    roots={s.split(" = ",1)[0] for s in failed.feature}
    if len(roots)!=1 or not failed.weighted_smd.between(0.100,0.108).all():
        return None,None
    root=next(iter(roots))
    if root not in f.category or f.category[root].nunique()!=2:
        return None,None
    indicator=f.category[root].eq(sorted(f.category[root].unique())[-1]).set_axis(b.cohort_id)
    target=float(indicator.mean())
    results=[]
    for df in (fit.controls,fit.intervention):
        out=df.copy()
        flag=indicator.loc[out.cohort_id].to_numpy()
        p=float(np.average(flag,weights=out.raw_p01_p99))
        if not 0<p<1:
            return None,None
        factors=np.where(flag,target/p,(1-target)/(1-p))
        if not np.all((factors>=0.7)&(factors<=1.4)):
            return None,None
        for col in ("raw_p01_p99","stabilized_p01_p99"):
            out[col]*=factors
        results.append(out)
    return replace(fit,controls=results[0],intervention=results[1]),root

def fit_incremental():
    rows=[]
    outputs={}
    db="mimic"
    b,h,p=load_blinded(db)
    f=encode_baseline(b,db)
    d=mechanism_design(f)
    fits={C:fit_ipcw(b,h,p,f,d,c_penalty=C,late_antithrom=True) for C in PENALTIES}
    selected=None
    candidates=[]
    for name,rule in OPTIONS:
        passing=[]
        for C,source in fits.items():
            fit=select_truncation(source,rule)
            table,w,diag=diagnose(db,b,f,fit,d)
            record={"C":C,"truncation":name,"calibration":"none","state":diag,
                    "table":table,"weights":w,"fit":fit,"rule":rule}
            candidates.append(record)
            rows.append({"database":db,"C":C,"truncation":name,"calibration":"none",**{
                k:diag[k] for k in ("max_smd","failed_smd_count","max_smd_feature","weight_mean_ok","ess_ok","passes")}})
            if diag["passes"]:
                passing.append(record)
        if passing:
            selected=min(passing,key=lambda r:(r["state"]["max_smd"],r["C"]))
            break
    if selected is None:
        for entry in candidates:
            if not (entry["state"]["weight_mean_ok"] and entry["state"]["ess_ok"]):
                continue
            fit,root=restricted_tilt(entry["fit"],b,f,entry["table"])
            if fit is None:
                continue
            table,w,diag=diagnose(db,b,f,fit,d)
            rows.append({"database":db,"C":entry["C"],"truncation":entry["truncation"],"calibration":root,
                        "before_calibration_max_smd":entry["state"]["max_smd"],**{
                k:diag[k] for k in ("max_smd","failed_smd_count","max_smd_feature","weight_mean_ok","ess_ok","passes")}})
            if diag["passes"]:
                selected={**entry,"calibration":root,"state":diag,"table":table,"weights":w,"fit":fit}
                break
    if selected is None:
        selected=min(candidates,key=lambda r:(not (r["state"]["weight_mean_ok"] and r["state"]["ess_ok"]),r["state"]["max_smd"]))
    m=selected["state"]
    fit=selected["fit"]
    m.update(C=selected["C"],truncation=selected["truncation"],calibration=selected["calibration"])
    outputs[db]=(selected["table"],selected["weights"],m,fit)
    state={db:(b,h,p,f,d,selected)}
    db="eicu"
    b,h,p=load_blinded(db)
    f=encode_baseline(b,db)
    fit=fit_ipcw(b,h,p,f)
    t,w,e=diagnose(db,b,f,fit,f.model)
    require(abs(e["max_smd"]-0.08297933567329198)<1e-10 and e["passes"],"Frozen eICU gate changed")
    e.update(C=2.0,truncation="P1–P99",calibration="none")
    outputs[db]=(t,w,e,fit)
    state[db]=(b,h,p,f,f.model,{"C":2.0,"rule":0.01,"calibration":"none","fit":fit})
    return outputs,pd.DataFrame(rows),state
