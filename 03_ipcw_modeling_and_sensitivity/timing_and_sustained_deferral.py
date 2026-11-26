from sensitivity_common import *

def describe_post_grace():
    lines=[]
    for db in DBS:
        b=pd.read_parquet(processed(db,"baseline"),columns=["cohort_id"]).sort_values("cohort_id")
        p=pd.read_parquet(processed(db,"ccw_clone_outcomes"),columns=["cohort_id","arm","eligible_g_cross_sectional_flag","artificial_censored","liberal_completed_by_g"])
        w=pd.read_parquet(REFERENCE/"weights"/f"{db}_locked_main.parquet",columns=["cohort_id","arm","w_after_g"])
        x=p.merge(w,on=["cohort_id","arm"],validate="one_to_one")
        x=x[x.eligible_g_cross_sectional_flag & ((x.arm.eq(0)&x.artificial_censored.eq(0)) |
                                                 (x.arm.eq(1)&x.liberal_completed_by_g))]
        r=pd.read_parquet(SENSITIVITY_INPUTS/f"{db}_rbc.parquet").sort_values("hr")
        hb=pd.read_parquet(SENSITIVITY_INPUTS/f"{db}_hb.parquet").sort_values(["hr","pref"],ascending=[True,False])
        def summaries(table,v,weights):
            order=np.argsort(v); vv=v[order]; ww=weights[order]
            mean=np.average(vv,weights=ww)
            q=vv[np.searchsorted(np.cumsum(ww)/ww.sum(),[.25,.5,.75])].tolist()
            return dict(**table,q1=q[0],median=q[1],q3=q[2],mean=mean,
                        sd=np.sqrt(np.average((vv-mean)**2,weights=ww)))

        for arm,z in x.groupby("arm"):
            denom=len(z)
            z=z.set_index("cohort_id")
            for lo,hi in [(6,24),(24,72),(6,168),(0,24),(0,168)]:
                events=r[r.hr.between(lo,hi,inclusive="both" if lo==0 else "right") & r.cohort_id.isin(z.index)]
                counts=events.groupby("cohort_id").size().reindex(z.index,fill_value=0)
                hit=counts.gt(0)
                core=dict(database=db,arm=arm,window=f"{lo}–{hi} h",denominator=mask(denom))
                lines.append(dict(**core,metric="Any recorded pRBC",n=mask(hit.sum()),pct=prop(hit.sum(),denom),
                                    ipcw_pct=100*np.average(hit,weights=z.w_after_g)))
                if lo==0:
                    for name,v in [("pRBC records per clone",counts.to_numpy())]:
                        for wt,q in [("unweighted",np.ones(len(z))),("IPCW",z.w_after_g.to_numpy())]:
                            lines.append(summaries(dict(**core,metric=name,weight=wt),v,q))
                    # Missing volume among recipients stays missing. Nonrecipients have an observed total of zero.
                    volume=events.groupby("cohort_id").ml.sum(min_count=1).reindex(z.index)
                    unknown=events.groupby("cohort_id").ml.apply(lambda s:s.isna().any()).reindex(z.index,fill_value=False)
                    volume.loc[~hit]=0
                    known=volume.notna()&~unknown
                    if known.sum()>=10:
                        for wt,q in [("unweighted",np.ones(known.sum())),("IPCW",z.w_after_g[known].to_numpy())]:
                            line=summaries(dict(database=db,arm=arm,window=f"{lo}–{hi} h",denominator=mask(known.sum()),
                                metric="Known complete pRBC volume (mL), including observed zero",weight=wt),volume[known].to_numpy(),q)
                            lines.append(line)
            if arm==0:
                events=r[r.hr.between(6,72,inclusive="right")&r.cohort_id.isin(z.index)]
                first=events.groupby("cohort_id",as_index=False).hr.min()
                joined=pre_hb(first,hb,12).join(z.w_after_g,on="cohort_id")
                bins=pd.cut(joined.hb,[-np.inf,7,8,9,np.inf],right=False,labels=["<7.0","7.0 to <8.0","8.0 to <9.0","≥9.0"])
                bins=bins.astype("string").fillna("No Hb in prior 12 h")
                for label in ["<7.0","7.0 to <8.0","8.0 to <9.0","≥9.0","No Hb in prior 12 h"]:
                    hit=bins.eq(label)
                    lines.append(dict(database=db,arm=0,window="6–72 h",metric="Pretransfusion Hb, "+label,
                         n=mask(hit.sum()),denominator=mask(len(hit)),pct=prop(hit.sum(),len(hit)),
                         ipcw_pct=100*np.average(hit,weights=joined.w_after_g) if len(hit) else np.nan))
                testing=joined.dropna(subset=["hb"])
                if len(testing)>=10:
                    for wt,q in [("unweighted",np.ones(len(testing))),("IPCW",testing.w_after_g.to_numpy())]:
                        lines.append(summaries(dict(database=db,arm=0,window="6–72 h",denominator=mask(len(testing)),
                                  metric="Measured pretransfusion Hb (g/dL)",weight=wt),testing.hb.to_numpy(),q))
    pd.DataFrame(lines).to_csv(TABLES/"post_grace_transfusion_and_pretransfusion_hb.csv",index=False)

def pre_hb(events,hb,tolerance):
    pat=hb[["cohort_id","hr","hb","pref"]].rename(columns={"hr":"hb_hr"})
    pat=pat.sort_values(["hb_hr","pref"],ascending=[True,False])
    return pd.merge_asof(events.sort_values("hr"),pat,left_on="hr",right_on="hb_hr",by="cohort_id",
                         allow_exact_matches=False,direction="backward",tolerance=float(tolerance))

def timing_and_feasibility():
    from calendar_and_subgroups import metric_rows
    timing=[]; sustained=[]
    for db in DBS:
        b=pd.read_parquet(processed(db,"baseline"),columns=["t0_from_icu_admit_hr","observed_strategy"])
        timing_bins=pd.cut(b.t0_from_icu_admit_hr,[-np.inf,6,12,24,48,120,np.inf],
                       labels=["≤6 h",">6–12 h",">12–24 h",">24–48 h",">48–120 h",">120 h"])
        for band in timing_bins.cat.categories:
            s=b[timing_bins.eq(band)]
            ev=s.observed_strategy.eq("early").sum()
            timing.append(dict(database=db,section="T0 distribution",scenario=band,n=mask(len(s)),pct=prop(len(s),len(b)),
                               early_count=mask(ev),early_pct=prop(ev,len(s))))
        for label in ("S_T0_le48h","S_T0_le24h","S_24h","S_48h"):
            status=SCENARIO_WEIGHTS/f"{db}_{label}.json"
            if not status.exists():
                continue
            s=json.loads(status.read_text())
            is_sustained=label in ("S_24h","S_48h")
            dest=sustained if is_sustained else timing
            attrs=dict(database=db,scenario=label)
            dest.extend(dict(**attrs,section="Outcome-free weight diagnostics",gate_pass=s["passes"],max_smd=s["max_smd"],
                        sustained_feasibility_pass=s["sustained_feasible"] if is_sustained else None,
                        **{**arm,"retained":mask(arm["retained"])}) for arm in s["arms"])
            frozen=SCENARIO_WEIGHTS/f"{db}_{label}.pkl"
            if is_sustained and s["passes"] and frozen.exists():
                t=load(frozen)["design"]
                for hour in range(t.g):
                    eligible=t.h0==hour
                    dest.append(dict(**attrs,section="Restrictive hourly diagnostics",arm=0,hour_from_t0=hour,
                               risk_count=mask(eligible.sum()),artificial_censors=mask(t.y0[eligible].sum())))
            path=WORK/f"boot_{db}_{label}.pkl"
            if path.exists():
                d=load(path); usable=len(d["rep"])
                dest += list(metric_rows(d["point"]["metrics"],
                   [v["metrics"] for v in d["rep"]] if usable>=.95*d["requested"] else [],
                   **attrs,section="Primary mortality estimates",n=mask(d["n"]),requested=d["requested"],valid=usable,
                   reps_with_bounded_factor=sum(q.get("tilt_bounded",0)>0 for q in d["rep"])))
    pd.DataFrame(timing).to_csv(TABLES/"t0_timing_and_acute_window_outcomes.csv",index=False)
    pd.DataFrame(sustained).to_csv(TABLES/"sustained_deferral_feasibility_and_estimates.csv",index=False)
