from cohort_database import *
from event_features import prepare_events,LAB_RANGE,VITAL_RANGE
from clinical_features import prepare_clinical,CHRONIC
from eicu import is_gib

LAB_BASE = tuple(x for x in LAB_RANGE if x!="hb")
CONTINUOUS = ("age","t0_from_icu_admit_hr","baseline_hb","hb_prior_drop_rate",
  "hr_last","hr_max_6h","sbp_last","sbp_min_6h","map_last","map_min_6h","rr_last",
  "spo2_min_6h","shock_index_t0","platelets","inr","pt","ptt","creatinine","bun",
  "bun_cr_ratio","wbc","bicarbonate","lactate","albumin","bilirubin")
CATEGORICAL = ("sex","race_eth","admit_source","icu_unit_type","off_hours_t0",
  "hb_bin_05","risk_phenotype","variceal_cirrhosis_portal_flag","gib_site","pud_flag",
  *CHRONIC,"has_prior_hb_flag","inr_pt_proxy_flag","lactate_measured_flag","vaso_any_t0",
  "vaso_dose_tier_t0","vent_status_t0","resp_charted_flag","ppi_h2ra_pre_t0",
  "antiplatelet_anticoag_pre_t0","endoscopy_confirmed_pre_t0","endoscopy_same_day_uncertain_flag")
COVER = (*LAB_BASE,"hr_last","sbp_last","map_last","rr_last","spo2_min_6h",
         "vaso_charted_flag","vaso_any_t0","resp_charted_flag")

def last(col,kind):
    return f"""arg_max({col},struct_pack(observed:=x.t, priority:=-x.pref, available:=x.known))
                    FILTER(WHERE x.kind={lit(kind)})"""

def points(k):
    build(f"{k}_points",f"""
      SELECT b.id,b.cohort,b.t0,b.hb_time,b.hb, hours*60.0 rel,b.t0+greatest(hours,0)*60.0 asof_min,
         hours hour_idx, CASE WHEN hours=-2 THEN 0 ELSE b.lab_start END lab_lower,
         CASE WHEN hours=-2 THEN 0 ELSE b.vital_start END vital_lower
      FROM {k}_bounds b CROSS JOIN LATERAL range(-2,cast(b.grace/60 AS BIGINT)) w(hours)
      WHERE hours<0 OR b.t0+hours*60 < b.exit_min
    """)
    lab_list=",".join(f"{last('x.val',kind)} AS {kind}" for kind in LAB_BASE)
    build(f"{k}_lp",f"""
      SELECT b.id,b.cohort,b.hour_idx,{lab_list},
         {last('x.val','hb')} observed_hb, {last('x.t','hb')} observed_hb_t,
         {last('x.known','hb')} known_hb_t,max(x.known) lab_latest_known
      FROM {k}_points b LEFT JOIN {k}_labe x ON x.id=b.id
         AND x.t>=b.lab_lower AND x.known<=b.asof_min
         AND ((b.hour_idx<0 AND (x.kind<>'hb' OR x.t<b.hb_time)) OR
              (b.hour_idx>=0 AND x.kind='hb' AND x.t>b.hb_time))
      GROUP BY b.id,b.cohort,b.hour_idx
    """)
    v_list=",".join(f"{last('x.val',kind)} AS {kind}_last" for kind in VITAL_RANGE)
    build(f"{k}_vp5",f"""
      WITH point_vital AS (
         SELECT b.id,b.cohort,b.hour_idx,x.kind,x.t,x.known,x.val,x.pref,b.t0
         FROM {k}_points b JOIN {k}_vitale x ON x.id=b.id
               AND x.t>=b.vital_lower AND x.known<=b.asof_min
         QUALIFY row_number() OVER(PARTITION BY b.id,b.cohort,b.hour_idx,x.kind,x.t
                   ORDER BY x.pref,x.known DESC,x.val)=1
      )
      SELECT x.id,x.cohort,x.hour_idx,{v_list},
         max(val) FILTER(WHERE kind='hr') hr_max_6h,
         min(val) FILTER(WHERE kind='sbp') sbp_min_6h,
         min(val) FILTER(WHERE kind='map') map_min_6h,
         min(val) FILTER(WHERE kind='spo2') spo2_min_6h,
         coalesce(bool_or(x.t>x.t0 AND ((kind='sbp' AND val<90) OR (kind='map' AND val<65))),false)
             new_hypotension_since_t0, max(x.known) vital_latest_known
      FROM point_vital x GROUP BY x.id,x.cohort,x.hour_idx
    """)
    if k=="m":
        build("m_vaso5","""
           SELECT b.id,b.cohort,b.hour_idx,
            count(x.id)>0 vaso_charted_flag,coalesce(bool_or(x.active),false) vaso_any_t0,
            CASE WHEN NOT coalesce(bool_or(x.active),false) THEN 0
                 WHEN max(x.nee) FILTER(WHERE x.active) IS NULL THEN 3
                 WHEN max(x.nee) FILTER(WHERE x.active)>=0.10 THEN 2 ELSE 1 END vaso_dose_tier_t0,
            max(x.t) vaso_latest_known
           FROM m_points b LEFT JOIN m_press_intervals x ON x.id=b.id AND x.t<=b.asof_min
                AND x.stop_t>b.asof_min AND x.stop_t>=CASE WHEN b.hour_idx=-2 THEN 0 ELSE b.asof_min-360 END
           GROUP BY b.id,b.cohort,b.hour_idx
        """)
    else:
        build("e_vaso5","""
           WITH active_records AS (
              SELECT b.id,b.cohort,b.hour_idx,x.nee,x.active,x.t,x.agent,x.pref
              FROM e_points b JOIN e_press_events x ON x.id=b.id AND x.t<=b.asof_min
                AND x.t>=CASE WHEN b.hour_idx=-2 THEN 0 ELSE b.asof_min-360 END
              QUALIFY row_number() OVER(PARTITION BY b.id,b.cohort,b.hour_idx,x.agent
                       ORDER BY x.pref,x.t DESC,x.record_id DESC)=1
           ), filtered AS (
             SELECT * FROM active_records
             QUALIFY agent<>'nonspecific' OR count(*) FILTER(WHERE agent<>'nonspecific' AND pref=1)
                     OVER(PARTITION BY id,cohort,hour_idx)=0
           )
           SELECT b.id,b.cohort,b.hour_idx,
              count(x.id)>0 vaso_charted_flag,coalesce(bool_or(x.active),false) vaso_any_t0,
              CASE WHEN NOT coalesce(bool_or(x.active),false) THEN 0
                   WHEN count(*) FILTER(WHERE x.active AND x.nee IS NULL)>0 THEN 3
                   WHEN sum(x.nee) FILTER(WHERE x.active)>=0.10 THEN 2 ELSE 1 END vaso_dose_tier_t0,
              max(x.t) vaso_latest_known
           FROM e_points b LEFT JOIN filtered x USING(id,cohort,hour_idx)
           GROUP BY b.id,b.cohort,b.hour_idx
        """)
    build(f"{k}_resp5",f"""
       SELECT b.id,b.cohort,b.hour_idx,count(x.id)>0 resp_charted_flag,
              coalesce(arg_max(x.val,struct_pack(observed:=x.t,priority:=-x.pref,val:=x.val)),0) vent_status_t0,
              max(x.known) resp_latest_known
       FROM {k}_points b LEFT JOIN {k}_resp x ON x.id=b.id AND x.known<=b.asof_min
           AND x.t>=CASE WHEN b.hour_idx=-2 THEN 0 ELSE b.asof_min-360 END
       WHERE b.hour_idx<0 GROUP BY b.id,b.cohort,b.hour_idx
    """)
    build(f"{k}_covariate_windows",f"""
        SELECT b.*,v.* EXCLUDE(id,cohort,hour_idx),l.* EXCLUDE(id,cohort,hour_idx),
           p.* EXCLUDE(id,cohort,hour_idx),s.* EXCLUDE(id,cohort,hour_idx)
        FROM {k}_points b LEFT JOIN {k}_vp5 v USING(id,cohort,hour_idx)
        LEFT JOIN {k}_lp l USING(id,cohort,hour_idx)
        LEFT JOIN {k}_vaso5 p USING(id,cohort,hour_idx)
        LEFT JOIN {k}_resp5 s USING(id,cohort,hour_idx)
    """)
    require(one(f"""SELECT count(*) FILTER(WHERE greatest(lab_latest_known,vital_latest_known,
                 vaso_latest_known,resp_latest_known)>asof_min) FROM {k}_covariate_windows""")[0]==0,
                f"{k} future information in feature set")

def baseline(k):
    demos=f"""d.sex,d.race_eth,d.icu_unit_type,
               CASE WHEN regexp_matches(lower(src),'emergency|^ed$') THEN 'ED'
                    WHEN regexp_matches(lower(src),'transfer from hospital|other hosp|outside hosp') THEN 'outside_hospital'
                    WHEN regexp_matches(lower(src),'operating|^or$|pacu|recovery|surgery') THEN 'OR_PACU'
                    WHEN regexp_matches(lower(src),'floor|ward|acute care|step.down') THEN 'ward' ELSE 'other_or_unknown' END admit_source"""
    if k=="m":
        src="Coalesce(case when b.ed_link OR d.has_ed THEN 'ED' END,d.admission_location)"
        t0date="(b.entry_at+b.t0*INTERVAL '1 minute')"
        dx_fields="""false::BOOLEAN eicu_acute_acs_flag,false::BOOLEAN eicu_chronic_cad_flag,
              NULL::BOOLEAN varlc_pre_t0, NULL::VARCHAR eicu_gib_site_24h_do_not_adjust_main"""
        joins="LEFT JOIN m_comorb g USING(hosp) LEFT JOIN m_comorb_pre mpre USING(id,cohort)"
        ap="NULL::DOUBLE"
        off=f"extract(hour from {t0date})<7 OR extract(hour from {t0date})>=19"
        endo=f"""coalesce(m.endoscopy,false) OR EXISTS (SELECT 1 FROM m_day_endoscopy p
                              WHERE p.id=b.id AND p.procedure_day<{t0date}::DATE)"""
        uncertain=f"EXISTS(SELECT 1 FROM m_day_endoscopy p WHERE p.id=b.id AND p.procedure_day={t0date}::DATE)"
        hosp="NULL::BIGINT"
    else:
        src="d.raw_source"
        dx_fields="g.eicu_acute_acs_flag,g.eicu_chronic_cad_flag,g.varlc_pre_t0, xsite.gib_site_24h eicu_gib_site_24h_do_not_adjust_main"
        joins="""LEFT JOIN e_prehist g USING(id,cohort)
                 LEFT JOIN e_apache_isolated x USING(id) LEFT JOIN e_dominance y USING(hospital)
                 LEFT JOIN e_site24 xsite USING(id)"""
        ap="x.apache_iv_do_not_adjust_main"
        off="extract(hour from (d.entry_clock+b.t0*INTERVAL '1 minute'))<7 OR extract(hour from (d.entry_clock+b.t0*INTERVAL '1 minute'))>=19"
        endo="coalesce(m.endoscopy,false)"
        uncertain="false"
        hosp="b.hospital"
    build(f"{k}_medications_at_baseline",f"""
       SELECT b.id,b.cohort,bool_or(x.acid) acid,bool_or(x.antithrom) antithrom,bool_or(x.endoscopy) endoscopy
       FROM {k}_bounds b LEFT JOIN {k}_drug_endo x ON x.id=b.id AND x.known<=b.t0
           AND x.t>=b.history_start GROUP BY b.id,b.cohort
    """)
    extra = """CASE WHEN b.tx_at IS NULL THEN 'none'
                    WHEN EXISTS(SELECT 1 FROM e_rbc r WHERE r.id=b.id AND r.t=b.tx_at AND r.channel='intakeOutput')
                     THEN 'io_with_ml' ELSE 'treatment_only' END eicu_prbc_capture_source,
               y.rate hosp_io_dominance_rate, y.rate>=0.80 hosp_io_dominant_ge80,
               inf.first_at IS NOT NULL eicu_infusiondrug_prbc_check,
               CASE WHEN least(b.tx_at,inf.first_at) IS NULL THEN b.observed_strategy
                    WHEN b.low_at < least(b.tx_at,inf.first_at) THEN 'restrictive'
                    ELSE 'early' END strategy_w1_6h_with_infdrug""" if k=="e" else """
                    NULL::VARCHAR eicu_prbc_capture_source,NULL::DOUBLE hosp_io_dominance_rate,
                    NULL::BOOLEAN hosp_io_dominant_ge80,NULL::BOOLEAN eicu_infusiondrug_prbc_check,
                    NULL::VARCHAR strategy_w1_6h_with_infdrug"""
    if k=="e":
        joins+="""
           LEFT JOIN LATERAL (SELECT min(t) first_at FROM e_rbc_inf r WHERE r.id=b.id
             AND r.t BETWEEN b.t0 AND least(b.t0+b.grace,b.exit_min)) inf ON TRUE
        """
    hist="mpre" if k=="m" else "g"
    confounders=",".join(f"coalesce({hist}.{name},false) {name}" for name in CHRONIC)
    if k=="m":
        confounders+=","+",".join(f"coalesce(g.{name},false) {name}_index_icd_do_not_adjust_main" for name in CHRONIC)
        confounders+=",coalesce(mpre.prior_hospital_history_flag,false) prior_hospital_history_flag"
    else:
        confounders+=",NULL::BOOLEAN prior_hospital_history_flag"
    build(f"{k}_baseline",f"""
       SELECT b.id AS cohort_id,b.cohort,{hosp} hospitalid,b.age,{demos.replace('src',src)},
          b.t0/60.0 t0_from_icu_admit_hr,({off}) off_hours_t0, false weekend_identifiable_flag,
          b.t0>=1440 t0_ge_24h_flag, {ap} apache_iv_do_not_adjust_main,
          CASE WHEN b.ihd AND b.shock THEN 'IHD_and_Shock'
               WHEN b.ihd THEN 'IHD_only' ELSE 'Shock_only' END risk_phenotype,
          {dx_fields},b.varlc variceal_cirrhosis_portal_flag,NOT b.varlc sub_non_variceal,
          CASE WHEN g.upper AND g.lower_gib THEN 'mixed'
               WHEN g.upper THEN 'upper' WHEN g.lower_gib THEN 'lower' ELSE 'unspecified' END gib_site,
          coalesce(g.pud_flag,false) pud_flag,{confounders},
          b.hb baseline_hb, b.hb_source, b.hb_time-b.t0 hb_anchor_at_rel_min,
          printf('%.1f',floor((b.hb-7)*2)/2+7) hb_bin_05,
          w.observed_hb IS NOT NULL has_prior_hb_flag,
          (w.observed_hb-b.hb)/nullif((b.hb_time-w.observed_hb_t)/60.0,0) hb_prior_drop_rate,
          w.hr_last,w.hr_max_6h,w.sbp_last,w.sbp_min_6h,w.map_last,w.map_min_6h,
          w.rr_last,w.spo2_min_6h,w.hr_last/nullif(w.sbp_last,0) shock_index_t0,
          {','.join('w.'+name for name in LAB_BASE)},
          w.inr IS NULL AND w.pt IS NOT NULL inr_pt_proxy_flag,
          CASE WHEN w.inr IS NOT NULL THEN 'INR' WHEN w.pt IS NOT NULL THEN 'PT_seconds' END coagulation_source,
          w.bun/nullif(w.creatinine,0) bun_cr_ratio, w.lactate IS NOT NULL lactate_measured_flag,
          w.vaso_charted_flag,w.vaso_any_t0,w.vaso_dose_tier_t0,
          w.vent_status_t0,w.resp_charted_flag,coalesce(m.acid,false) ppi_h2ra_pre_t0,
          coalesce(m.antithrom,false) antiplatelet_anticoag_pre_t0,
          ({endo}) endoscopy_confirmed_pre_t0,({uncertain}) endoscopy_same_day_uncertain_flag,
          b.observed_strategy,b.obs_rescue compliant_rescue_lt7,b.tie_hb7_tx_flag,{extra}
       FROM {k}_bounds b JOIN {k}_demo d USING(id) JOIN {k}_covariate_windows w USING(id,cohort)
          LEFT JOIN {k}_medications_at_baseline m USING(id,cohort) {joins}
       WHERE w.hour_idx=-1
    """)
    require(n(f"{k}_baseline")==n(f"{k}_trial_stays"),f"{k} baseline count drift")

def make_features(k):
    prepare_events(k)
    prepare_clinical(k)
    points(k)
    baseline(k)
    print(f"{k}: baseline and time-varying snapshots assembled and timestamp checks passed.",flush=True)
