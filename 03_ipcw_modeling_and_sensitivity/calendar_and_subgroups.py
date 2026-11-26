from sensitivity_models import *
import argparse

PRIMARY={"mimic":"28-day all-cause mortality","eicu":"28-day in-hospital mortality (live discharge retained)"}
METRICS=[("risk_restrictive_pct",0,100,0),("risk_liberal_pct",1,100,0),("rd_pp",2,1,0),("rr",3,1,1),("rmst_difference_days",4,1,0)]

def metric_rows(point,reps,defs=METRICS,**attrs):
    points=np.asarray(point)
    samples=np.asarray(reps)
    for name,k,scale,null in defs:
        if k>=len(points):
            continue
        info=tail(samples[:,k]*scale if samples.ndim==2 else [],null)
        if name.startswith("risk_") or name in ("mean_restrictive_days","mean_liberal_days"):
            info={k:v for k,v in info.items() if k in ("lo","hi","boot_n")}
        yield dict(**attrs,metric=name,estimate=points[k]*scale,**info)

def main_export():
    outcome=[]; groups=[]; hb=[]; frames=[]; trends=[]
    for db in DBS:
        if db=="mimic":
            with connect() as con:
                b,_,_=projections(con,db)
            for era,g in b.groupby("anchor_year_group"):
                early=g.observed_strategy.eq("early").sum(); wait=g.observed_strategy.eq("restrictive").sum()
                shifted=pd.read_parquet(SENSITIVITY_INPUTS/"mimic_baseline_extensions.parquet").set_index("cohort_id").loc[g.cohort_id]
                trends.append(dict(anchor_year_group=era,n=len(g),early=mask(early),early_pct=prop(early,len(g)),
                    observed_deferral=mask(wait),observed_deferral_pct=prop(wait,len(g)),
                    baseline_hb_mean=g.baseline_hb.mean(),shifted_icu_year_differs_from_anchor=mask(
                    (shifted.anchor_year!=shifted.shifted_icu_year).sum())))
        for label in ("Main","S_2014_2022","S_2011_2022"):
            file=WORK/f"boot_{db}_{label}.pkl"
            if not file.exists():
                continue
            d=load(file); p=d["point"]; r=d["rep"]
            attrs=dict(database=db,scenario=label,n=mask(d["n"]),frame=d["frame"],requested=d["requested"],valid=len(r),
                       reps_with_bounded_factor=sum(q.get("tilt_bounded",0)>0 for q in r))
            # A result with too few valid replicates stays available as a diagnostic.
            estimation_status="complete" if len(r)>=.95*d["requested"] else "insufficient valid replicates"
            attrs["inference_status"]=estimation_status
            r=r if estimation_status=="complete" else []
            outcome.extend(metric_rows(p["metrics"],[q["metrics"] for q in r],
                endpoint=PRIMARY[db],events_restrictive=mask(p["events"][0]),events_liberal=mask(p["events"][1]),**attrs))
            if label!="Main":
                continue
            outcome.extend(metric_rows(p["cox"],[q["cox"] for q in r],
                  [(k,i,1,1) for i,k in enumerate(("hr_total","hr_0_to_7d","hr_after_7_to_28d"))],endpoint=PRIMARY[db],**attrs))
            for endpoint,stat in p["secondary"].items():
                outcome.extend(metric_rows(stat[:4],[q["secondary"][endpoint][:4] for q in r],endpoint=endpoint,
                     events_restrictive=mask(stat[4]),events_liberal=mask(stat[5]),**attrs))
            outcome.extend(metric_rows(p["free"],[q["free"] for q in r],
                   [(k,i,1,0) for i,k in enumerate(("mean_restrictive_days","mean_liberal_days","mean_difference_days"))],
                   endpoint="ICU-free days through day 28",**attrs))
            for name,stat in p["groups"].items():
                level_names={
                     "Ischemic heart disease":("IHD present","Shock without IHD"),
                     "Varices, cirrhosis or portal hypertension":("Variceal phenotype","Nonvariceal"),
                     "Baseline Hb 7.0 to <8.0 g/dL":("Hb 7.0 to <8.0","Hb 8.0 to <9.0"),
                     "Age ≥75 years":("Age ≥75","Age <75"),
                     "Vasopressor use at time zero":("Vasopressor","No vasopressor"),
                     "Acute ACS vs chronic CAD":("Acute ACS/MI","Chronic CAD without acute ACS/MI"),
                     "Strict instability (hypotension or vasopressors)":("Strict instability","Without strict instability")}[name]
                for i,level in enumerate(level_names):
                    sizes=dict(attrs)
                    sizes["n"]=mask(d["group_n"][name][i])
                    groups.extend(metric_rows(stat[i],[q["groups"][name][i] for q in r],domain=name,level=level,**sizes))
                groups.extend(metric_rows(stat[2],[q["groups"][name][2] for q in r],
                              [("interaction_rd_pp",2,1,0),("interaction_log_rr",3,1,0)],domain=name,level="interaction",**attrs))
            for i in range(20):
                hb += list(metric_rows(p["hb"][:,i],[q["hb"][:,i] for q in r],hb_g_dl=(70+i)/10,**attrs))
            if db=="eicu":
                original=load(REFERENCE/"runtime"/"bootstrap"/"eicu_Main.pkl")
                frames.extend(metric_rows(original["point"]["metrics"],[r["metrics"] for r in original["replicates"]],
                                          database=db,method="Reference IPCW with all logging-active hospitals",frame=160,requested=500))
                frames.extend(metric_rows(p["metrics"],[q["metrics"] for q in r],database=db,
                                          method="Enrolling hospital frame",frame=d["frame"],requested=d["requested"]))
                other=WORK/"boot_eicu_Main160.pkl"
                if other.exists():
                    prior=load(other)
                    frames.extend(metric_rows(prior["point"]["metrics"],[q["metrics"] for q in prior["rep"]],
                         database=db,method="All logging-active hospitals with revised IPCW",frame=160,requested=prior["requested"]))
    for name,items in [
        ("mimic_calendar_era_trends",trends),("calendar_adjusted_main_outcomes",outcome),
        ("subgroup_estimates",groups),("hemoglobin_effect_curve",hb),("eicu_hospital_bootstrap_frames",frames)
    ]:
        if items:
            pd.DataFrame(items).to_csv(TABLES/f"{name}.csv",index=False)
    balances=[]
    gates=[]
    for file in sorted(SCENARIO_WEIGHTS.glob("*.json")):
        st=json.loads(file.read_text())
        db,label=st["database"],st["scenario"]
        keys={k:v for k,v in st.items() if k not in ("arms","calibrated")}
        for arm in st["arms"]:
            gates.append({**keys,**arm,"retained":mask(arm["retained"]),"calibrated":" | ".join(st["calibrated"])})
        balances.append(pd.read_csv(WORK/f"balance_{db}_{label}.csv").assign(gate_pass=st["passes"]))
    if balances:
        pd.concat(balances,ignore_index=True).to_csv(TABLES/"covariate_balance.csv",index=False)
        pd.DataFrame(gates).to_csv(TABLES/"weight_model_diagnostics.csv",index=False)
