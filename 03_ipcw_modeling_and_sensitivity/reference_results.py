from settings import *
from bootstrap_trials import BOOT_CACHE,SCENARIOS
from weight_validation import require_unblinding_authorization
import json
import pickle
import numpy as np
import pandas as pd

DATABASE={"mimic":"MIMIC-IV v3.1","eicu":"eICU-CRD"}
PRIMARY={"mimic":"28-day all-cause mortality","eicu":"28-day in-hospital mortality (discharge retained)"}
SENS={
 "S1":"Narrow Hb window 7.0 to <8.5 g/dL","S2":"Four-hour grace window","S3":"ED Hb up to two hours before ICU",
 "S4":"Hourly liberal initiation model","S5a":"No weight truncation","S5b":"P0.5–P99.5 truncation","S5c":"P2.5–P97.5 truncation",
 "S6":"Immediate ICU transfer censoring","S7":"Include index-stay chronic ICD codes",
 "S8a":"IntakeOutput-dominant hospitals","S8b":"Add infusionDrug RBC pump records",
 "S9-MIMIC":"Precisely timed in-hospital death within 28 days","S9-eICU":"T0 ≥24 h with APACHE IV",
}
GROUPS={
 "Ischemic heart disease":("IHD present","Shock without IHD"),
 "Varices, cirrhosis or portal hypertension":("Varices/cirrhosis/portal hypertension","Nonvariceal high-risk GIB"),
 "Baseline Hb 7.0 to <8.0 g/dL":("Hb 7.0 to <8.0 g/dL","Hb 8.0 to <9.0 g/dL"),
 "Age ≥75 years":("Age ≥75 years","Age <75 years"),
 "Vasopressor use at time zero":("Vasopressor present","No vasopressor"),
 "Acute ACS vs chronic CAD":("Acute ACS/MI proxy","Chronic CAD proxy without acute ACS/MI"),
}

def finite(x):
    x=np.asarray(x,dtype=float)
    return x[np.isfinite(x)]

def ci(x):
    x=finite(x)
    return tuple(np.quantile(x,[.025,.975])) if len(x)>1 else (np.nan,np.nan)

def pval(x, null):
    x=finite(x)
    if len(x)<=1:
        return np.nan
    left=((x<=null).sum()+1)/(len(x)+1)
    right=((x>=null).sum()+1)/(len(x)+1)
    return min(1,2*min(left,right))

def ef(x):
    if not np.isfinite(x) or x<=0:
        return np.nan
    y=max(x,1/x)
    return y+np.sqrt(y*(y-1))

def effect(values,reps,rmst=True):
    data={}
    defs=[("risk_restrictive_pct",0,100),("risk_liberal_pct",1,100),("rd_pp",2,1),("rr",3,1)]
    if rmst:
        defs.append(("rmst_difference_days",4,1))
    for name,i,scale in defs:
        val=values[i]*scale
        rep=np.asarray(reps)[:,i]*scale
        lo,hi=ci(rep)
        data.update({name:val,name+"_l95":lo,name+"_u95":hi,name+"_n":len(finite(rep))})
        if i>=2:
            data[name+"_p"]=pval(rep,1 if i==3 else 0)
    return data

def read_cache(db,label):
    with (BOOT_CACHE/f"{db}_{label}.pkl").open("rb") as f:
        q=pickle.load(f)
    expected=500 if label=="Main" else 200
    require(q["database"]==db and q["scenario"]==label and q["attempts"]==expected and len(q["replicates"])>=.95*expected,"Invalid bootstrap checkpoint")
    return q

def collect():
    require_unblinding_authorization()
    stats=[]; groups=[]; sens=[]; curve_rows=[]; hb_rows=[]; methods=[]
    for db in EXPECTED:
        d=read_cache(db,"Main"); point=d["point"]; rep=d["replicates"]
        e=effect(point["metrics"],[q["metrics"] for q in rep])
        rr=e["rr"]; lo=e["rr_l95"]; hi=e["rr_u95"]
        e["e_value_rr"]=ef(rr)
        e["e_value_nearest_ci"]=1.0 if lo<=1<=hi else ef(lo if rr>=1 else hi)
        rd=e["rd_pp"]
        e["nnt_or_nnh"]=100/abs(rd) if rd!=0 else np.nan
        e["nnt_or_nnh_type"]="NNH" if rd>0 else "NNT" if rd<0 else "Undefined"
        low_rd,high_rd=e["rd_pp_l95"],e["rd_pp_u95"]
        if low_rd*high_rd>0:
            e["nnt_or_nnh_l95"],e["nnt_or_nnh_u95"]=sorted((100/abs(low_rd),100/abs(high_rd)))
        e.update({"database":db,"endpoint":PRIMARY[db],"role":"primary","n":d["n"],"boot_valid":len(rep),
                  "events_restrictive":masked(point["events"][0]),"events_liberal":masked(point["events"][1])})
        for k,label in enumerate(("hr_total","hr_0_to_7d","hr_after_7_to_28d")):
            samples=np.array([r["cox"][k] for r in rep])
            limits=ci(samples)
            e.update({label:point["cox"][k],label+"_l95":limits[0],label+"_u95":limits[1],label+"_p":pval(samples,1),
                      label+"_n":len(finite(samples))})
        stats.append(e)
        for endpoint,sm in point["secondary"].items():
            ep=effect(sm,[r["secondary"][endpoint] for r in rep],rmst=False)
            stats.append({"database":db,"endpoint":endpoint,"role":"secondary","n":d["n"],"boot_valid":len(rep),
                          "events_restrictive":masked(sm[4]),"events_liberal":masked(sm[5]),**ep})
        free=point["free"]
        values=np.array([x["free"] for x in rep])
        temp={"database":db,"endpoint":"ICU-free days through day 28","role":"secondary","n":d["n"],"boot_valid":len(rep)}
        for j,label in enumerate(("mean_restrictive_days","mean_liberal_days","mean_difference_days")):
            temp[label]=free[j]
            temp[label+"_l95"],temp[label+"_u95"]=ci(values[:,j])
            temp[label+"_n"]=len(finite(values[:,j]))
        temp["mean_difference_days_p"]=pval(values[:,2],0)
        stats.append(temp)
        for dim,sm in point["groups"].items():
            reps=np.stack([r["groups"][dim] for r in rep])
            for j,sub in enumerate(GROUPS[dim]):
                efc=effect(sm[j],reps[:,j,:])
                groups.append({"database":db,"domain":dim,"level":sub,"n":masked(d["group_sizes"][dim][j]),
                               "interaction_rd_pp":sm[2,2],"interaction_rd_l95":ci(reps[:,2,2])[0],
                               "interaction_rd_u95":ci(reps[:,2,2])[1],"interaction_p_rd":pval(reps[:,2,2],0),
                               "interaction_log_rr":sm[2,3],"interaction_log_rr_l95":ci(reps[:,2,3])[0],
                               "interaction_log_rr_u95":ci(reps[:,2,3])[1],"interaction_p_log_rr":pval(reps[:,2,3],0),
                               **efc})
        for code,cohort,rule in SCENARIOS[db]:
            s=read_cache(db,code)
            d0=s["point"]; reps=s["replicates"]
            sens.append({"database":db,"test":code,"description":SENS[code],"cohort":cohort,"n":s["n"],
                         "boot_valid":len(reps),"events_restrictive":masked(d0["events"][0]),
                         "events_liberal":masked(d0["events"][1]),**effect(d0["metrics"],[q["metrics"] for q in reps])})
            methods.append([db,code,s["n"],s["cluster_frame"],s["clusters_represented"],len(reps),s["attempts"],
                            sum(r.get("tilt_bounded",0)>0 for r in reps)])
        methods.append([db,"Main",d["n"],d["cluster_frame"],d["clusters_represented"],len(rep),d["attempts"],
                        sum(r.get("tilt_bounded",0)>0 for r in rep)])
        cur=np.stack([x["curve"] for x in rep])
        l,h=np.quantile(cur,[.025,.975],axis=0)
        for a in (0,1):
            curve_rows.append({"database":db,"arm":a,"hour":0,"cumulative_risk":0.0,"lower95":0.0,"upper95":0.0,
                               "event_free":1.0,"at_risk":masked(d["n"])})
            for j in range(1,673):
                k=j-1
                idx=min(j,671)
                curve_rows.append({"database":db,"arm":a,"hour":j,"cumulative_risk":point["curve"][a,k],"lower95":l[a,k],
                    "upper95":h[a,k],"event_free":1-point["curve"][a,k],"at_risk":masked(point["risk"][a,idx])})
        hboot=np.stack([r["hb"] for r in rep])
        for j,hb in enumerate(np.arange(70,90)/10):
            row={"database":db,"hb_g_dl":hb}
            for k,name in enumerate(("risk_restrictive","risk_liberal","rd_pp","rr")):
                row[name]=point["hb"][k,j]
                row[name+"_l95"],row[name+"_u95"]=ci(hboot[:,k,j])
                row[name+"_n"]=len(finite(hboot[:,k,j]))
            hb_rows.append(row)
    return list(map(pd.DataFrame,(stats,groups,sens,curve_rows,hb_rows))),pd.DataFrame(methods,columns=[
        "database","scenario","n","clusters_resampled","clusters_with_observations","n_valid","n_requested","n_calibration_factor_clipped"])

def csv(df,dst):
    floatcols=df.select_dtypes("number").columns
    if not df[floatcols].empty:
        df=df.copy()
        df[floatcols]=df[floatcols].replace([np.inf,-np.inf],np.nan)
    df.to_csv(dst,index=False,float_format="%.10g",na_rep="Not estimable")
