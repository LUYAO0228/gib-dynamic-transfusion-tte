from settings import *
from dataclasses import dataclass
from model_features import encode_baseline, standardized, ns
from weight_selection import mechanism_design
from weight_model_inputs import TIME_VARYING, CHRONIC
from weight_validation import require_unblinding_authorization
import numpy as np
import pandas as pd
import duckdb

@dataclass
class Trial:
    db: str
    label: str
    g: int
    n: int
    x: np.ndarray
    x0: np.ndarray
    i0: np.ndarray
    h0: np.ndarray
    y0: np.ndarray
    a1: np.ndarray
    y1: np.ndarray
    landmark: np.ndarray
    a0: np.ndarray
    c: float
    rule: object
    tilt: object
    xlib: object
    methods: str
    art: np.ndarray
    primary_time: np.ndarray
    primary_event: np.ndarray
    hosp_rel: np.ndarray
    hosp_death: np.ndarray
    icu_rel: np.ndarray
    icu_death: np.ndarray
    free: np.ndarray
    secondary: dict
    group: dict
    hb_fit: np.ndarray
    hb_predict: np.ndarray
    clusters: np.ndarray
    cluster_frame: int
    hb: np.ndarray

from config import ec, raw, ts

class AuthorizedRead:
    def __init__(self):
        self.status=require_unblinding_authorization()
        self.con=duckdb.connect(config={"temp_directory":str(RUNTIME),"threads":3})
    def pq(self, db, table, cohort):
        return self.con.execute("SELECT * FROM read_parquet(?)",
            [str(PROCESSED/f"{db}_{table}_{cohort}.parquet")]).df()
    def extras(self,db,b):
        self.con.register("ids",b[["cohort_id","t0_from_icu_admit_hr"]])
        try:
            if db=="mimic":
                out=self.con.execute(f"""
                    SELECT b.cohort_id,date_diff('second',{ts('i.intime')},{ts('i.outtime')})/3600.0-b.t0_from_icu_admit_hr icu_rel
                    FROM {raw('icu','icustays')} i JOIN ids b ON i.stay_id=b.cohort_id
                    """).df()
            else:
                out=self.con.execute(f"""
                    SELECT b.cohort_id,p.unitdischargeoffset/60.0-b.t0_from_icu_admit_hr icu_rel
                    FROM {ec('patient')} p JOIN ids b ON p.patientunitstayid=b.cohort_id
                    """).df()
        finally:
            self.con.unregister("ids")
        require(len(out)==len(b) and out.cohort_id.is_unique,"ICU exit timing join failed")
        return out.set_index("cohort_id").icu_rel.reindex(b.cohort_id).to_numpy()
    def additions(self,b,icu):
        from pathlib import Path
        self.con.register("trial_window",b[["cohort_id","t0_from_icu_admit_hr"]].assign(icu_rel=icu))
        is_rbc=r"volume-transfuse rbc|volume-transfuse unmatched o-neg rbc|^\s*prbc|^\s*rbc|^packed (red blood )?cells|^blood \(prbcs\)"
        exclude=r"non.prbc|ns for blood|flush|ffp|plasma|platelet|cryo"
        try:
            x=self.con.execute(f"""
              SELECT cohort_id, min(j.infusionoffset-b.t0_from_icu_admit_hr*60) AS infusion_min
              FROM {ec('infusiondrug')} j JOIN trial_window b ON j.patientunitstayid=b.cohort_id
              WHERE j.infusionoffset-b.t0_from_icu_admit_hr*60 BETWEEN -0.00001 AND least(360,b.icu_rel*60)
                AND regexp_matches(lower(j.drugname),?) AND NOT regexp_matches(lower(j.drugname),?)
                AND greatest(try_cast(replace(j.drugrate,',','') AS DOUBLE),j.infusionrate)>0
              GROUP BY cohort_id
              """,[is_rbc,exclude]).df()
        finally:
            self.con.unregister("trial_window")
        return x.set_index("cohort_id").infusion_min.reindex(b.cohort_id).clip(lower=0).to_numpy()
    def frozen(self,db):
        return self.con.execute("SELECT * FROM read_parquet(?)",[str(WEIGHTS/f"{db}_locked_main.parquet")]).df()

def reshape(o,col):
    return np.vstack([o[a][col].to_numpy() for a in (0,1)])

def add_infusion(b,h,o,first,icu):
    a0,a1=o
    old=a1.first_prbc_rel_min.to_numpy(dtype=float)
    first=np.where(np.isnan(first),np.inf,first)
    old=np.where(np.isnan(old),np.inf,old)
    earlier=first<old
    nontransfused=b.observed_strategy.eq("restrictive").to_numpy() & ~np.isfinite(old)
    require(int((nontransfused & earlier).sum())==15,"InfusionDrug reclassification count differs from the reference count")
    low=a0.rescue_trigger_rel_min.fillna(np.inf).to_numpy()
    unrestricted=earlier&~(low<first)
    ix=np.flatnonzero(earlier)
    times=dict(zip(b.cohort_id.iloc[ix],first[ix]))
    h=h.copy()
    t=h.cohort_id.map(times).fillna(np.inf)
    hour=np.maximum(np.ceil(t/60)-1,0)
    control=h.arm.eq(0)
    newly_artificial=h.cohort_id.isin(b.cohort_id.loc[unrestricted])
    h=h.loc[~((control & newly_artificial | ~control & t.lt(np.inf)) & h.hour_idx.gt(hour))].copy()
    t=t.loc[h.index]; hour=hour.loc[h.index]
    control=h.arm.eq(0)
    newly_artificial=newly_artificial.loc[h.index]
    h.loc[control & newly_artificial,"early_prbc_censor_event"]=h.loc[control & newly_artificial,"hour_idx"].eq(hour).astype("int32")
    h.loc[~control & t.lt(np.inf),"liberal_prbc_completion"]=h.loc[~control & t.lt(np.inf),"hour_idx"].eq(hour).astype("int32")
    h.loc[~control & t.lt(np.inf),"live_exit_censor_event"]=0
    a0.loc[unrestricted,"artificial_censored"]=1
    a0.loc[unrestricted,"censor_time_min"]=first[unrestricted]
    a1.loc[earlier,"liberal_completed_by_g"]=True
    a1.loc[earlier,"artificial_censored"]=0
    a1.loc[earlier,"artificial_censored_if_exit_at_g"]=False
    a1.loc[earlier,"censor_time_if_exit_at_g_min"]=360
    return h

def load_trial(source,db,label="Main",cohort=MAIN,rule=None):
    b=source.pq(db,"baseline",cohort).sort_values("cohort_id").reset_index(drop=True)
    h=source.pq(db,"ccw_grace_person_hour",cohort)
    o=source.pq(db,"ccw_clone_outcomes",cohort)
    frame=160 if db=="eicu" else len(b)
    if label=="S8a":
        b=b.loc[b.hosp_io_dominant_ge80].reset_index(drop=True)
        require(len(b)==2137 and b.hospitalid.nunique()==102,"S8 dominant-IO cohort drift")
        frame=102
    elif label=="S9-eICU":
        b=b.loc[b.t0_ge_24h_flag & b.apache_iv_do_not_adjust_main.notna()].reset_index(drop=True)
        require(len(b)==330,"S9 APACHE eligible cohort drift")
    elif label=="S7":
        for k in CHRONIC:
            b[k]=b[k].fillna(False)|b[k+"_index_icd_do_not_adjust_main"].fillna(False)
    if db=="mimic":
        frame=len(b)
    keys=pd.Series(np.arange(len(b)),index=b.cohort_id)
    h=h.loc[h.cohort_id.isin(keys.index)].copy()
    o=[o.loc[o.arm.eq(a)].set_index("cohort_id").reindex(keys.index).reset_index() for a in (0,1)]
    g=int(o[0].grace_min.iloc[0]//60)
    require(all(c.grace_min.eq(g*60).all() for c in o),"Cohort and clone key mismatch")
    icu=source.extras(db,b)
    if label=="S8b":
        first=source.additions(b,icu)
        h=add_infusion(b,h,o,first,icu)
    f=encode_baseline(b,db)
    base=mechanism_design(f) if db=="mimic" else f.model.copy()
    if label=="S9-eICU":
        base["apache_iv_s9"]=standardized(b.apache_iv_do_not_adjust_main.astype(float))
    x=base.to_numpy()
    def step(rows, late=False):
        i=keys.loc[rows.cohort_id].to_numpy()
        hour=rows.hour_idx.to_numpy(int)
        z=rows[list(TIME_VARYING)].astype(float).copy()
        z["sbp_locf"]=np.where(z.sbp_locf.isna(),f.values.sbp_last.to_numpy()[i],z.sbp_locf)
        for name in ("hb_change_from_t0","sbp_locf"):
            z[name]=standardized(z[name])
        hour_bin=(hour[:,None]==np.arange(1,g)[None,:]).astype(float)
        extra=[(b.antiplatelet_anticoag_pre_t0.to_numpy(dtype=bool)[i]&(hour>=3)).astype(float)] if late else []
        return i,hour,np.column_stack([x[i],z,hour_bin,*extra])
    a0=h.loc[h.arm.eq(0)&h.at_risk_for_censor.eq(1)].sort_values(["cohort_id","hour_idx"])
    i0,h0,x0=step(a0,db=="mimic")
    y0=a0.early_prbc_censor_event.to_numpy(int)
    one=o[1]
    elig=one.liberal_completed_by_g|one.artificial_censored_if_exit_at_g
    if label=="S6":
        elig=one.liberal_completed_by_g|one.censor_reason.eq("missed_deadline")
    dbconfig=source.status["databases"][db]
    if rule is None:
        rule=0.01
    xlib=None
    if label in ("S4","S6"):
        l=h.loc[h.arm.eq(1)]
        if label=="S6":
            l=l.loc[l.at_risk_for_exit_censor.eq(1)]
        l=l.sort_values(["cohort_id","hour_idx"])
        i,kh,design=step(l)
        target=l.liberal_prbc_completion if label=="S4" else l.live_exit_censor_event
        xt=None
        if label=="S4":
            # Predict each patient's no-initiation path from the last observed
            # pre-initiation history carried forward after the last observed interval.
            arr=np.empty((len(b),g,design.shape[1]))
            arr[:]=np.nan
            arr[i,kh,:]=design
            for j in range(1,g):
                absent=np.isnan(arr[:,j,0])
                arr[absent,j,:]=arr[absent,j-1,:]
            arr[:,:,-(g-1):]=(np.arange(g)[:,None]==np.arange(1,g)[None,:])[None,:,:]
            xt=arr.reshape(-1,design.shape[1])
        xlib=(i,kh,design,target.to_numpy(int),xt)
    tilt=b.antiplatelet_anticoag_pre_t0.to_numpy(bool) if dbconfig["calibration"]!="none" else None
    a0_elig=o[0].eligible_g_cross_sectional_flag.to_numpy(bool)&o[0].artificial_censored.eq(0).to_numpy()
    a1_elig=one.eligible_g_cross_sectional_flag.to_numpy(bool)&one.liberal_completed_by_g.to_numpy(bool)
    landmark=np.vstack((a0_elig,a1_elig))
    c0=np.where(o[0].artificial_censored.eq(1),o[0].censor_time_min/60,np.inf)
    col,tcol=("artificial_censored","censor_time_min") if label=="S6" else ("artificial_censored_if_exit_at_g","censor_time_if_exit_at_g_min")
    c1=np.where(one[col].astype(bool),one[tcol]/60,np.inf)
    art=np.vstack((c0,c1))
    hosp=o[0].hosp_observation_rel_min.to_numpy(float)/60
    hosp_death=o[0].death_inhosp.to_numpy(bool)
    evt=o[0].primary_death.to_numpy(bool)
    tm=o[0].primary_obs_min.to_numpy(float)/60
    if db=="eicu":
        # ICU deaths during the grace period are natural
        # terminal observations. Use the documented ICU death time when
        # ward/hospital administrative discharge was posted later.
        g_icu_death=o[0].icu_death.to_numpy(bool)&hosp_death&(icu<=g)&(icu<hosp)
        require(np.all(evt[g_icu_death]),"Grace ICU death diverges from the 28-day endpoint indicator")
        tm=np.where(g_icu_death,icu,tm)
        hosp=np.where(g_icu_death,icu,hosp)
    if label=="S9-MIMIC":
        evt=evt & hosp_death & ~o[0].dod_date_only_flag.to_numpy(bool)
        tm=np.where(evt,tm,672.)
    secondary={
        "In-hospital mortality (entire index stay)":(hosp,hosp_death,None),
        "ICU mortality":(np.where(o[0].icu_death,np.minimum(icu,hosp),icu),o[0].icu_death.to_numpy(bool),None),
        "Delayed RBC retransfusion":(np.where(o[0].delayed_retransfusion_gt24h,o[0].delayed_retransfusion_rel_min/60,
                                        hosp if db=="mimic" else icu),o[0].delayed_retransfusion_gt24h.to_numpy(bool),None),
    }
    if db=="mimic":
        u=o[0].death_union_28d_inhosp.to_numpy(bool)
        union=np.where(o[0].death_28d,tm,np.where(hosp_death,hosp,np.maximum(hosp,672)))
        secondary.update({
            "28-day or in-hospital death":(union,u,None),
            "New measured troponin T elevation":(np.where(o[0].lab_new_tnt_elevation_post_t0,o[0].tnt_first_event_rel_min/60,hosp),o[0].lab_new_tnt_elevation_post_t0.to_numpy(bool),None),
            "New troponin T elevation or ICD-coded MI":(o[0].tnt_first_event_rel_min.fillna(o[0].hosp_observation_rel_min).to_numpy()/60,
                    o[0].composite_tnt_or_icd_mi.to_numpy(bool),None),
        })
    else:
        secondary["28-day in-hospital death (discharge-censored KM)"]=(np.minimum(hosp,672),evt,672)
    group={"Ischemic heart disease":b.risk_phenotype.ne("Shock_only").to_numpy(),
           "Varices, cirrhosis or portal hypertension":b.variceal_cirrhosis_portal_flag.to_numpy(bool),
           "Baseline Hb 7.0 to <8.0 g/dL":b.baseline_hb.lt(8).to_numpy(),
           "Age ≥75 years":b.age.ge(75).to_numpy(),
           "Vasopressor use at time zero":b.vaso_any_t0.to_numpy(bool)}
    if db=="eicu":
        ac=b.eicu_acute_acs_flag.to_numpy(bool)
        ch=b.eicu_chronic_cad_flag.to_numpy(bool)
        group["Acute ACS vs chronic CAD"]=(ac, ch & ~ac)
    hb=ns(b.baseline_hb.astype(float),2)
    from patsy import dmatrix,build_design_matrices
    design_h=dmatrix("cr(hb, df=2, constraints='center')-1",{"hb":b.baseline_hb})
    new=build_design_matrices([design_h.design_info],{"hb":np.arange(70,90)/10})[0]
    clusters=pd.factorize(b.hospitalid)[0] if db=="eicu" else np.arange(len(b))
    require((clusters>=0).all() and max(clusters)<frame,"Invalid bootstrap cluster IDs")
    require(np.isfinite(hosp).all() and np.isfinite(icu).all(),"Missing verified hospital or ICU duration")
    return Trial(db,label,g,len(b),x,x0,i0,h0,y0,elig.to_numpy(),one.liberal_completed_by_g.to_numpy(int),
        landmark,a0_elig,dbconfig["C"],rule,tilt,xlib,label,art,tm,evt,hosp,hosp_death,icu,
        o[0].icu_death.to_numpy(bool),o[0].icu_free_days_28d.to_numpy(),secondary,group,
        np.asarray(design_h),np.asarray(new),clusters,frame,b.baseline_hb.to_numpy())
