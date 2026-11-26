from sensitivity_models import *
from sklearn.linear_model import PoissonRegressor
from causal_estimators import frozen_weights
from trial_data import load_trial
import itertools

import calendar_and_subgroups as outcome_tables
from timing_and_sustained_deferral import pre_hb

def retransfusion_data(t,b):
    db=t.db
    r=pd.read_parquet(SENSITIVITY_INPUTS/f"{db}_rbc.parquet")
    key=pd.Series(np.arange(len(b)),index=b.cohort_id)
    r=r[r.cohort_id.isin(key.index)&r.hr.gt(24)&r.hr.le(168)].copy()
    r["j"]=r.cohort_id.map(key)
    r=r[r.hr.le(t.hosp_rel[r.j])]
    any_first=r.groupby("cohort_id").hr.min()
    rolled={}
    for key_i,s in r[r.ml.notna()].groupby("cohort_id"):
        s=s.sort_values("hr")
        tt=s.hr.to_numpy()
        x=np.r_[0,np.cumsum(s.ml)]
        total=x[1:]-x[np.searchsorted(tt,tt-24,side="right")]
        ok=tt[total>=600]
        if len(ok):
            rolled[key_i]=ok[0]
    eligible=(t.hosp_rel>24)&~(t.primary_event&(t.primary_time<=24))
    endpoint={
       "exploratory_rbc_any_24h_to_7d":(eligible,any_first.reindex(b.cohort_id).fillna(np.inf).to_numpy()),
       "exploratory_rbc_ge600ml_24h_to_7d":(eligible,pd.Series(rolled,dtype=float).reindex(b.cohort_id).fillna(np.inf).to_numpy())}
    hb=pd.read_parquet(SENSITIVITY_INPUTS/f"{db}_hb.parquet")
    fp=pre_hb(any_first.rename("hr").reset_index(),hb,192).set_index("cohort_id").hb.reindex(b.cohort_id)
    return endpoint,fp

def retransfusion(db, workers, scenario):
    if not (SCENARIO_WEIGHTS/f"{db}_{scenario}.json").exists():
        return
    item=unblind(db,scenario)
    method="revised analysis frozen weights"
    if item is None and scenario=="Main":
        # Use the reference IPCW weights when the revised main weights are unavailable.
        src=AuthorizedRead()
        try:
            t=load_trial(src,db)
            b=src.pq(db,"baseline",COHORT).sort_values("cohort_id").reset_index(drop=True)
            w=frozen_weights(src.frozen(db),t)
        finally:
            src.con.close()
        t.cluster_frame=len(np.unique(t.clusters))
        method="reference analysis frozen weights, revised analysis enrolling hospital frame"
    elif item is None:
        return
    else:
        t,w,b=item
    t.label="retransfusion_"+scenario
    endpoint,hb=retransfusion_data(t,b)
    if db=="eicu":
        endpoint={k:v for k,v in endpoint.items() if ("ge600" in k)==(scenario=="eicu_intake_output_dominant")}
    result=run_boot(t,w,500,workers,False,WORK/f"boot_retransfusion_{db}_{scenario}.pkl",endpoint)
    lines=[]
    for name,(elig,etime) in endpoint.items():
        r=result["point"][name]
        usable=len(result["rep"])
        attr=dict(database=db,scenario=scenario,endpoint=name,weights=method,n=mask(t.n),requested=500,valid=usable,
             eligible_restrictive=mask(r["denominator"][0]),eligible_liberal=mask(r["denominator"][1]),
             events_restrictive=mask(r["events"][0]),events_liberal=mask(r["events"][1]))
        lines.extend(outcome_tables.metric_rows(r["metrics"],[q[name]["metrics"] for q in result["rep"]] if usable>=475 else [],
                      outcome_tables.METRICS[:4],**attr,section="landmark competing-risk estimate"))
        if "any" in name:
            for arm in (0,1):
                selected=elig&np.isfinite(etime)&(t.art[arm]>=etime)
                v=hb.to_numpy()[selected]
                for key,flag in [
                    ("Pretransfusion Hb <7",v<7),("Pretransfusion Hb 7 to <8",(v>=7)&(v<8)),
                    ("Pretransfusion Hb ≥8",v>=8),("Pretransfusion Hb unmeasured",~np.isfinite(v))]:
                    lines.append(dict(**attr,section="pretransfusion Hb",arm=arm,metric=key,
                          count=mask(flag.sum()),denominator=mask(len(v)),pct=prop(flag.sum(),len(v)),
                          ipcw_pct=100*np.average(flag,weights=w.late[arm,selected]) if len(v) else np.nan))
    pd.DataFrame(lines).to_csv(WORK/f"retransfusion_{db}_{scenario}.csv",index=False)

def evalue(ratio):
    if not np.isfinite(ratio) or ratio<=0:
        return np.nan
    value=max(ratio,1/ratio)
    return value+np.sqrt(value*(value-1))

def qba():
    result=[]
    for db in DBS:
        path=WORK/f"boot_{db}_Main.pkl"
        d=load(path) if path.exists() else load(REFERENCE/"runtime"/"bootstrap"/f"{db}_Main.pkl")
        reps=d.get("rep",d.get("replicates"))
        method="revised analysis" if path.exists() else "reference weights (revised diagnostics did not pass)"
        rr=d["point"]["metrics"][3]
        stat=tail([x["metrics"][3] for x in reps],1)
        lo=stat["lo"]
        for target in (.90,.85,.80):
            ratio=rr/target
            result.append(dict(database=db,type="non-null E-value",reference=method,rr_observed=rr,rr_lower95=lo,
                 rr_true_target=target,bias_factor=ratio,e_value_point=evalue(ratio) if ratio>1 else 1.,
                 e_value_ci=evalue(lo/target) if lo>target else 1.))
        for eu,ud in itertools.product((1.5,2,2.5,3,4),repeat=2):
            bf=eu*ud/(eu+ud-1)
            result.append(dict(database=db,type="bias factor grid",reference=method,rr_observed=rr,rr_eu=eu,rr_ud=ud,
                               bias_factor=bf,adjusted_rr_under_max_upward_bias=rr/bf))
        estimates=[
            ("Days 0–7 hazard ratio",d["point"]["cox"][1],[x["cox"][1] for x in reps]),
            ("Variceal phenotype RR",d["point"]["groups"]["Varices, cirrhosis or portal hypertension"][0,3],
             [x["groups"]["Varices, cirrhosis or portal hypertension"][0,3] for x in reps])]
        s1=load(REFERENCE/"runtime"/"bootstrap"/f"{db}_S1.pkl")
        estimates.append(("Narrow Hb window RR (reference analysis)",s1["point"]["metrics"][3],
                          [x["metrics"][3] for x in s1["replicates"]]))
        for endpoint,value,samples in estimates:
            info=tail(samples,1)
            lo,hi=info["lo"],info["hi"]
            edge=1 if lo<=1<=hi else lo if value>=1 else hi
            result.append(dict(database=db,type="null E-value",reference=method,endpoint=endpoint,rr_observed=value,
                        rr_lower95=lo,rr_upper95=hi,e_value_point=evalue(value),e_value_ci=evalue(edge),
                        scale="HR treated as ratio; approximation" if "hazard" in endpoint else "RR"))
        with connect() as con:
            b,_,_=projections(con,db)
        b=b.sort_values("cohort_id").reset_index(drop=True)
        f=encode_baseline(b,db)
        outcome=pd.read_parquet(processed(db,"ccw_clone_outcomes"),columns=["cohort_id","arm","primary_death"])
        event=outcome[outcome.arm.eq(0)].set_index("cohort_id").primary_death.reindex(b.cohort_id).to_numpy()
        observed=b.observed_strategy.isin(["early","restrictive"]).to_numpy()
        y=b.observed_strategy.eq("early").to_numpy()
        full=pd.concat([f.values,pd.get_dummies(f.category,drop_first=True,dtype=float)],axis=1)
        for name,z,remove in [("Prior Hb decline ≥0.3 g/dL/h",b.hb_prior_drop_rate.ge(.3),"hb_prior_drop_rate"),
                              ("Baseline Hb <7.5 g/dL",b.baseline_hb.lt(7.5),"baseline_hb"),
                              ("Vasopressor at T0",b.vaso_dose_tier_t0.gt(0),"vaso_dose_tier_t0")]:
            keep=[k for k in full if not k.startswith(remove)]
            mat=np.column_stack([z.to_numpy(float),full[keep].apply(standardized)])
            for key,target,idx in [("Recorded early transfusion",y,observed),("28-day mortality",event,np.ones(len(b),bool))]:
                zz=z.to_numpy()[idx]
                yy=np.asarray(target[idx],dtype=float)
                dsg=mat[idx].copy()
                fit=PoissonRegressor(alpha=.01,max_iter=300).fit(dsg,yy)
                dsg[:,0]=0; p0=fit.predict(dsg).mean()
                dsg[:,0]=1; p1=fit.predict(dsg).mean()
                observed_rr=yy[zz].mean()/yy[~zz].mean()
                result.append(dict(database=db,type="observed benchmark",reference=method,endpoint=key,
                       feature=name,n=mask(len(yy)),exposed=mask(zz.sum()),rr_observed=observed_rr,
                       adjusted_standardized_rr=p1/p0,scale="Ridge Poisson covariate standardization; descriptive"))
            # The bias factor's exposure–confounder parameter compares marker
            # prevalence across exposure levels, so export that direction too.
            yes=y[observed]
            marker=z.to_numpy(float)[observed]
            act=np.column_stack([yes,mat[observed,1:]])
            fit=PoissonRegressor(alpha=.01,max_iter=300).fit(act,marker)
            act[:,0]=0; p0=fit.predict(act).mean()
            act[:,0]=1; p1=fit.predict(act).mean()
            result.append(dict(database=db,type="observed benchmark",reference=method,
                 endpoint="Marker prevalence, early versus deferral",feature=name,n=mask(len(yes)),
                 exposed=mask(yes.sum()),rr_observed=marker[yes].mean()/marker[~yes].mean(),
                 adjusted_standardized_rr=p1/p0,scale="Ridge Poisson covariate standardization; descriptive"))
    pd.DataFrame(result).to_csv(TABLES/"quantitative_bias_analysis.csv",index=False)

def export_retransfusion():
    files=sorted(WORK.glob("retransfusion_*_*.csv"))
    if files:
        pd.concat([pd.read_csv(f) for f in files],ignore_index=True).to_csv(
            TABLES/"exploratory_retransfusion_24h_to_7d.csv",index=False)
