from settings import *
import duckdb
import re

CONTINUOUS=("baseline_hb","t0_from_icu_admit_hr","age","sbp_min_6h",
            "hr_last","hr_max_6h","sbp_last","map_last","map_min_6h","rr_last","spo2_min_6h",
            "shock_index_t0","platelets","inr","creatinine","bun","bun_cr_ratio","wbc","bicarbonate",
            "hb_prior_drop_rate")
IMPUTE=CONTINUOUS[3:-1]
LAB_CATS=("lactate","albumin","bilirubin","ptt")
CHRONIC=("chf","af","ckd_esrd","copd","diabetes","malignancy")
BINARY=("off_hours_t0",*CHRONIC,"has_prior_hb_flag","resp_charted_flag","ppi_h2ra_pre_t0",
        "antiplatelet_anticoag_pre_t0","endoscopy_confirmed_pre_t0")
COHORT_SPECIFIC={
  "mimic":("variceal_cirrhosis_portal_flag","prior_hospital_history_flag","endoscopy_same_day_uncertain_flag"),
  "eicu":("varlc_pre_t0","eicu_acute_acs_flag","eicu_chronic_cad_flag"),
}
MULTI=("sex","race_eth","admit_source","icu_unit_type","risk_phenotype","vaso_dose_tier_t0","vent_status_t0")
AUX_BASE=("cohort_id","observed_strategy","pt","inr_pt_proxy_flag")
TIME_VARYING=("hb_change_from_t0","new_hb_measured_prev_hr","sbp_locf",
              "new_hypotension_since_t0","vaso_active_locf","non_rbc_blood_since_t0")
PANEL=("cohort_id","arm","hour_idx","at_risk_for_censor","early_prbc_censor_event",*TIME_VARYING)
PROTOCOL=("cohort_id","arm","artificial_censored","artificial_censored_if_exit_at_g",
          "liberal_completed_by_g","eligible_g_cross_sectional_flag")
FORBIDDEN=re.compile(r"death|apache|primary_event|troponin|tnt|retransfusion|icu_free_days|outcome_in_grace|analysis_followup",re.I)

def read_columns(con,path,columns):
    require(all(re.fullmatch(r"[a-z][a-z0-9_]*",x) and not FORBIDDEN.search(x) for x in columns),
            "Non-permitted variable in blinded Parquet projection")
    require(len(columns)==len(set(columns)),"Duplicate projected field")
    names=", ".join('"' + column + '"' for column in columns)
    return con.execute(f"SELECT {names} FROM read_parquet(?)",[str(path)]).df()

def load_blinded(db):
    require(db in EXPECTED,"Unknown database")
    cols=(*AUX_BASE,*CONTINUOUS,*LAB_CATS,*BINARY,*COHORT_SPECIFIC[db],*MULTI)
    with duckdb.connect(config={"threads":2,"temp_directory":str(RUNTIME)}) as conn:
        b=read_columns(conn,PROCESSED/f"{db}_baseline_{MAIN}.parquet",cols).sort_values("cohort_id").reset_index(drop=True)
        h=read_columns(conn,PROCESSED/f"{db}_ccw_grace_person_hour_{MAIN}.parquet",PANEL)
        c=read_columns(conn,PROCESSED/f"{db}_ccw_clone_outcomes_{MAIN}.parquet",PROTOCOL)
    n,hours,completers=EXPECTED[db]
    require(len(b)==n and b.cohort_id.is_unique and len(h)==hours and len(c)==2*n,
            "cohort and cloning counts or baseline IDs failed validation")
    require(not h.duplicated(["cohort_id","arm","hour_idx"]).any() and not c.duplicated(["cohort_id","arm"]).any(),
            "Duplicate clone or risk-set row")
    require(set(b.cohort_id)==set(h.cohort_id)==set(c.cohort_id),"Cohort ID mismatch")
    require(c.loc[c.arm.eq(1),"liberal_completed_by_g"].sum()==completers,"Liberal adherence count drift")
    return b,h,c
