from cohort_database import *

FOLLOWUP = 28*1440

def patient_outcomes(k):
    if k=="m":
        build("m_icu_occupancy",f"""
             SELECT b.id,b.cohort,greatest(b.t0, date_diff('second',b.entry_at,x.entry_at)/60.0) start_t,
                least(b.t0+{FOLLOWUP}, date_diff('second',b.entry_at,x.exit_at)/60.0) stop_t
             FROM m_trial_stays b JOIN m0 x ON b.person=x.person
             WHERE x.entry_at < b.entry_at+(b.t0+{FOLLOWUP})*INTERVAL '1 minute'
               AND x.exit_at > b.entry_at+b.t0*INTERVAL '1 minute'
        """)
        build("m_obs_out",f"""
           WITH troponin AS (
              SELECT b.id,b.cohort,
                count(*) FILTER(WHERE x.t<=b.t0 AND x.val<=x.ul)>0 prior_normal,
                count(*) FILTER(WHERE x.t<=b.t0 AND x.val>x.ul)>0 prior_high,
                min(x.t) FILTER(WHERE x.t>b.t0 AND x.val>x.ul) tnt_t
              FROM m_trial_stays b LEFT JOIN m_labs x ON b.id=x.id AND x.itemid='51003'
                 AND x.ul>0 AND x.val IS NOT NULL AND x.t<=b.discharge_min
              GROUP BY b.id,b.cohort
           ), tx_day AS (
              SELECT b.id,b.cohort,floor((x.t-b.t0)/1440) day_index,min(x.t) first_t,
                 max(x.t) last_t
              FROM m_trial_stays b JOIN m_rbc x USING(id)
              WHERE x.t>=b.t0+1440 AND x.t<=b.discharge_min
              GROUP BY ALL HAVING sum(x.units)>=2 OR sum(x.ml)>=600
           ), death AS (
              SELECT b.*,
                CASE WHEN death_min IS NOT NULL THEN death_min
                     WHEN dod IS NOT NULL AND dod>=(entry_at+t0*INTERVAL '1 minute')::DATE
                        THEN greatest(b.t0,coalesce(b.discharge_min,b.t0),
                              date_diff('second',entry_at,dod::TIMESTAMP+INTERVAL '12 hours')/60.0)
                END observed_death_min,
                CASE WHEN death_min IS NOT NULL THEN death_min BETWEEN t0 AND t0+{FOLLOWUP}
                     ELSE dod BETWEEN (entry_at+t0*INTERVAL '1 minute')::DATE
                               AND (entry_at+(t0+{FOLLOWUP})*INTERVAL '1 minute')::DATE
                END death_flag
              FROM m_trial_stays b
           )
           SELECT b.id,b.cohort,
             coalesce(b.death_flag,false) death_28d,
             CASE WHEN b.death_flag THEN least(b.observed_death_min-b.t0,{FOLLOWUP})/60.0 ELSE 672.0 END time_to_death_28d_hr,
             coalesce(b.hospitalized_death,false) death_inhosp,
             coalesce(b.hospitalized_death,false) OR coalesce(b.death_flag,false) death_union_28d_inhosp,
             (b.death_min IS NULL AND b.death_flag IS TRUE) dod_date_only_flag,
             coalesce(b.death_at BETWEEN b.entry_at+b.t0*INTERVAL '1 minute' AND b.exit_at,false) icu_death,
             coalesce(prior_normal AND NOT prior_high AND tnt_t IS NOT NULL,false) lab_new_tnt_elevation_post_t0,
             coalesce(b.ami OR (prior_normal AND NOT prior_high AND tnt_t IS NOT NULL),false) composite_tnt_or_icd_mi,
             CASE WHEN prior_normal AND NOT prior_high THEN tnt_t-b.t0 END tnt_first_event_rel_min,
             td.min_tx_day IS NOT NULL delayed_retransfusion_gt24h,
             td.min_tx_day-b.t0 delayed_retransfusion_rel_min,
             coalesce(b.death_flag,false) primary_death,
             CASE WHEN b.death_flag THEN least(b.observed_death_min-b.t0,{FOLLOWUP}) ELSE {FOLLOWUP} END primary_obs_min,
             least(b.discharge_min,coalesce(b.death_min,b.discharge_min))-b.t0 hosp_observation_rel_min,
             NULL::BOOLEAN death_inhosp_28d,NULL::DOUBLE time_to_death_inhosp_28d_hr
           FROM death b LEFT JOIN troponin t USING(id,cohort)
           LEFT JOIN (SELECT id,cohort,min(last_t) min_tx_day FROM tx_day GROUP BY id,cohort) td USING(id,cohort)
        """)
    else:
        build("e_icu_occupancy",f"""
             SELECT b.id,b.cohort,greatest(b.t0, b.adm_min-x.adm_min) start_t,
                least(b.t0+{FOLLOWUP}, b.adm_min-x.adm_min+x.exit_min) stop_t
             FROM e_trial_stays b JOIN e0 x ON b.hospital=x.hospital AND b.hosp=x.hosp
             WHERE b.adm_min-x.adm_min < b.t0+{FOLLOWUP} AND b.adm_min-x.adm_min+x.exit_min > b.t0
        """)
        build("e_obs_out",f"""
           WITH known AS (
              SELECT b.*,b.hospital_status IN ('expired','dead') hospital_death,
                     (SELECT min(x.t) FROM e_rbc x WHERE x.id=b.id
                       AND x.t BETWEEN b.t0+1440 AND b.exit_min) repeat_t
              FROM e_trial_stays b
           )
           SELECT id,cohort,NULL::BOOLEAN death_28d,NULL::DOUBLE time_to_death_28d_hr,
             coalesce(hospital_death,false) death_inhosp,NULL::BOOLEAN death_union_28d_inhosp,
             NULL::BOOLEAN dod_date_only_flag,coalesce(unit_status IN ('expired','dead'),false) icu_death,
             NULL::BOOLEAN lab_new_tnt_elevation_post_t0,NULL::BOOLEAN composite_tnt_or_icd_mi,
             NULL::DOUBLE tnt_first_event_rel_min,
             repeat_t IS NOT NULL delayed_retransfusion_gt24h,repeat_t-t0 delayed_retransfusion_rel_min,
             coalesce(hospital_death AND discharge_min BETWEEN t0 AND t0+{FOLLOWUP},false) primary_death,
             greatest(0,least(discharge_min-t0,{FOLLOWUP})) primary_obs_min,
             discharge_min-t0 hosp_observation_rel_min,
             coalesce(hospital_death AND discharge_min BETWEEN t0 AND t0+{FOLLOWUP},false) death_inhosp_28d,
             greatest(0,least(discharge_min-t0,{FOLLOWUP}))/60.0 time_to_death_inhosp_28d_hr
           FROM known
        """)
    build(f"{k}_icu_free",f"""
       WITH prev AS (
         SELECT *,max(stop_t) OVER (PARTITION BY id,cohort ORDER BY start_t,stop_t
                           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) before_end
         FROM {k}_icu_occupancy WHERE start_t<stop_t
       ), runs AS (
         SELECT *,sum(CASE WHEN before_end IS NULL OR start_t>before_end THEN 1 ELSE 0 END)
           OVER(PARTITION BY id,cohort ORDER BY start_t,stop_t ROWS UNBOUNDED PRECEDING) segment
         FROM prev
       ), totals AS (
         SELECT id,cohort,sum(duration_min) minutes_icu FROM (
           SELECT id,cohort,segment,max(stop_t)-min(start_t) AS duration_min FROM runs GROUP BY ALL
         ) GROUP BY id,cohort
       )
       SELECT o.id,o.cohort,CASE WHEN o.primary_death THEN 0.0 ELSE
              greatest(0,28-coalesce(t.minutes_icu,0)/1440.0) END icu_free_days_28d
       FROM {k}_obs_out o LEFT JOIN totals t USING(id,cohort)
    """)
    require(n(f"{k}_obs_out")==n(f"{k}_trial_stays"),f"{k} outcome join mismatch")

def assign_policy_paths(k):
    build(f"{k}_ccw_path",f"""
       WITH times AS (
         SELECT b.*,
            CASE WHEN death_min>t0 AND death_min<=t0+grace THEN death_min-t0 END early_death_rel,
            CASE WHEN exit_min>t0 AND exit_min<=t0+grace
                   AND (death_min IS NULL OR death_min>exit_min) THEN exit_min-t0 END live_exit_rel,
            tx_at-t0 tx_rel,low_at-t0 low_rel
         FROM {k}_trial_stays b
       ), qualified AS (
         SELECT *,
            CASE WHEN tx_rel IS NOT NULL AND (early_death_rel IS NULL OR tx_rel<early_death_rel)
                     AND (live_exit_rel IS NULL OR tx_rel<=live_exit_rel) THEN tx_rel END tx_while_at_risk,
            CASE WHEN low_rel IS NOT NULL AND
                   (early_death_rel IS NULL OR low_rel<early_death_rel) AND
                   (live_exit_rel IS NULL OR low_rel<=live_exit_rel)
                   AND (tx_rel IS NULL OR floor(round(low_at+minute_shift,8))<floor(round(tx_at+minute_shift,8)))
                 THEN low_rel END effective_low_rel
         FROM times
       ), policy AS (
         SELECT a.*,arms.arm,
             CASE WHEN arms.arm=1 THEN tx_while_at_risk ELSE effective_low_rel END compliance_rel,

             CASE WHEN arms.arm=0 AND effective_low_rel IS NULL THEN tx_while_at_risk
                  WHEN arms.arm=1 AND tx_while_at_risk IS NULL AND
                       (early_death_rel IS NULL OR (live_exit_rel IS NOT NULL AND live_exit_rel<early_death_rel))
                    THEN coalesce(live_exit_rel,grace) END corrected_artificial_rel
         FROM qualified a CROSS JOIN (VALUES(0),(1)) arms(arm)
       )
       SELECT * EXCLUDE(corrected_artificial_rel,compliance_rel),
          compliance_rel,corrected_artificial_rel artificial_rel,
          least(coalesce(early_death_rel,grace),coalesce(live_exit_rel,grace),
                coalesce(compliance_rel,grace),coalesce(corrected_artificial_rel,grace)) risk_stop_rel,
          CASE WHEN corrected_artificial_rel IS NULL THEN 'none'
               WHEN arm=0 THEN 'early_prbc'
               WHEN live_exit_rel IS NOT NULL AND tx_while_at_risk IS NULL THEN 'live_icu_exit'
               ELSE 'missed_deadline' END censor_reason
       FROM policy
    """)


def clone_paths(k):
    assign_policy_paths(k)
    build(f"{k}_ccw_outcomes",f"""
        SELECT b.id cohort_id, b.cohort,cast(b.id AS VARCHAR)||'_'||cast(arm AS VARCHAR) clone_id,arm,
           v.hospitalid,v.baseline_hb,v.hb_bin_05,v.risk_phenotype,v.variceal_cirrhosis_portal_flag,v.sub_non_variceal,
           grace grace_min,(artificial_rel IS NOT NULL)::INTEGER artificial_censored,
           coalesce(artificial_rel,grace) censor_time_min,censor_reason,
           CASE WHEN arm=1 AND censor_reason='live_icu_exit' THEN grace ELSE coalesce(artificial_rel,grace) END
             censor_time_if_exit_at_g_min,
           CASE WHEN arm=1 AND censor_reason='live_icu_exit' AND b.death_min<=b.t0+grace
                   THEN false ELSE artificial_rel IS NOT NULL END artificial_censored_if_exit_at_g,
           (arm=0 AND effective_low_rel IS NOT NULL AND tx_while_at_risk IS NOT NULL)::INTEGER compliant_rescue_lt7,
           b.tie_hb7_tx_flag,tx_while_at_risk first_prbc_rel_min,effective_low_rel rescue_trigger_rel_min,
           compliance_rel grace_compliance_rel_min, risk_stop_rel grace_weight_risk_stop_rel_min,
           early_death_rel IS NOT NULL AND (artificial_rel IS NULL OR early_death_rel<=artificial_rel)
               outcome_in_grace,
           live_exit_rel IS NOT NULL AND (tx_rel IS NULL OR tx_rel>live_exit_rel) left_icu_untransfused_in_grace,
           early_death_rel IS NULL AND live_exit_rel IS NULL eligible_g_cross_sectional_flag,
           tx_while_at_risk IS NOT NULL liberal_completed_by_g,
           o.* EXCLUDE(id,cohort),u.icu_free_days_28d,
           CASE WHEN o.composite_tnt_or_icd_mi IS NOT NULL THEN o.composite_tnt_or_icd_mi END troponin_mi_post_t0,
           least(o.primary_obs_min,artificial_rel)/60.0 analysis_followup_hr,
           o.primary_death AND (artificial_rel IS NULL OR o.primary_obs_min<=artificial_rel) analysis_primary_event,
           least(o.primary_obs_min,CASE WHEN artificial_censored_if_exit_at_g THEN censor_time_if_exit_at_g_min END)/60.0
                analysis_followup_if_exit_at_g_hr,
           o.primary_death AND (NOT artificial_censored_if_exit_at_g OR o.primary_obs_min<=censor_time_if_exit_at_g_min)
                analysis_primary_event_if_exit_at_g
        FROM {k}_ccw_path b JOIN {k}_baseline v ON b.id=v.cohort_id AND b.cohort=v.cohort
        JOIN {k}_obs_out o ON b.id=o.id AND b.cohort=o.cohort
        JOIN {k}_icu_free u ON b.id=u.id AND b.cohort=u.cohort
    """)

def grace_panel(k):
    build(f"{k}_ccw_hour",f"""
      WITH risk AS (
         SELECT b.id,b.cohort,b.arm,b.t0,b.grace,b.risk_stop_rel,b.early_death_rel,b.live_exit_rel,b.compliance_rel,
                b.artificial_rel,b.censor_reason,b.tx_while_at_risk,b.effective_low_rel,
                b.tie_hb7_tx_flag,b.hb_time,
                s.rel start_min,s.rel+60 stop_min,s.hour_idx,s.asof_min,s.observed_hb,s.hb,
                s.hr_last,s.sbp_last,s.map_last,s.new_hypotension_since_t0,s.vaso_any_t0,
                s.lab_latest_known,s.vital_latest_known,s.vaso_latest_known
         FROM {k}_ccw_path b JOIN {k}_covariate_windows s USING(id,cohort)
         WHERE s.hour_idx>=0 AND (s.rel<b.risk_stop_rel OR s.hour_idx=0)
      )
      SELECT id cohort_id,cohort,cast(id AS VARCHAR)||'_'||cast(arm AS VARCHAR) clone_id,arm,hour_idx,
            grace grace_min,start_min interval_start_min,stop_min interval_end_min,
            least(stop_min,risk_stop_rel) observed_interval_end_min,
            (CASE WHEN arm=0 THEN true ELSE stop_min=grace AND risk_stop_rel=grace END)::INTEGER at_risk_for_censor,
            (arm=1 AND tx_while_at_risk IS DISTINCT FROM 0)::INTEGER at_risk_for_exit_censor,
            (artificial_rel IS NOT NULL AND artificial_rel<=stop_min
                 AND (artificial_rel>start_min OR hour_idx=0))::INTEGER artificial_censor_event,
            (arm=0 AND censor_reason='early_prbc' AND artificial_rel<=stop_min
                 AND (artificial_rel>start_min OR hour_idx=0))::INTEGER early_prbc_censor_event,
            (arm=1 AND censor_reason='missed_deadline' AND artificial_rel=stop_min)::INTEGER deadline_censor_event,
            (arm=1 AND censor_reason='live_icu_exit' AND artificial_rel<=stop_min AND artificial_rel>start_min)::INTEGER live_exit_censor_event,
            (arm=0 AND compliance_rel IS NOT NULL AND compliance_rel>start_min AND compliance_rel<=stop_min)::INTEGER rescue_exemption_event,
            (arm=1 AND compliance_rel IS NOT NULL AND (compliance_rel>start_min OR hour_idx=0)
                 AND compliance_rel<=stop_min)::INTEGER liberal_prbc_completion,
            (early_death_rel IS NOT NULL AND early_death_rel=risk_stop_rel AND early_death_rel<=stop_min AND early_death_rel>start_min)::INTEGER outcome_in_grace,
            (live_exit_rel IS NOT NULL AND live_exit_rel=risk_stop_rel AND live_exit_rel<=stop_min
                 AND live_exit_rel>start_min AND (tx_while_at_risk IS NULL OR tx_while_at_risk>live_exit_rel))::INTEGER left_icu_untransfused_in_grace,
            coalesce(tx_while_at_risk=0 AND hour_idx=0,false) tx_at_t0_flag,tie_hb7_tx_flag,
            CASE WHEN hour_idx=0 THEN hb ELSE coalesce(observed_hb,hb) END hb_locf,
            CASE WHEN hour_idx=0 THEN 0.0 ELSE coalesce(observed_hb,hb)-hb END hb_change_from_t0,
            EXISTS(SELECT 1 FROM {k}_labe x WHERE x.id=risk.id AND x.kind='hb'
                   AND x.known>asof_min-60 AND x.known<=asof_min AND x.t>=hb_time) new_hb_measured_prev_hr,
            hr_last hr_locf,sbp_last sbp_locf,map_last map_locf,
            coalesce(new_hypotension_since_t0,false) new_hypotension_since_t0,
            vaso_any_t0 vaso_active_locf,
            EXISTS(SELECT 1 FROM {k}_blood r WHERE r.id=risk.id AND r.t>t0 AND r.known<=asof_min)
                non_rbc_blood_since_t0
      FROM risk
    """)

def verify_ccw(k):
    require(n(f"{k}_ccw_outcomes")==2*n(f"{k}_trial_stays"),f"{k} clone count mismatch")
    require(one(f"""
        SELECT count(*) FROM (
           SELECT cohort,cohort_id,count(*) n,count(DISTINCT arm) arms FROM {k}_ccw_outcomes GROUP BY ALL
           HAVING n<>2 OR arms<>2)
      """)[0]==0,f"{k} each participant must have two arms")
    require(one(f"""
        SELECT count(*) FROM (
          SELECT cohort,clone_id,count(*) n,max(hour_idx)+1 last_plus_one,count(DISTINCT hour_idx) hrs,
                 sum(artificial_censor_event) stops
          FROM {k}_ccw_hour GROUP BY ALL HAVING n<>last_plus_one OR n<>hrs OR stops>1)
      """)[0]==0,f"{k} noncontiguous hourly rows or repeated artificial censor")
    require(one(f"""
         SELECT count(*) FROM {k}_ccw_hour
         WHERE observed_interval_end_min<interval_start_min OR observed_interval_end_min>interval_end_min
            OR hour_idx*60>=grace_min OR (outcome_in_grace=1 AND artificial_censor_event=1)
            OR (arm=0 AND early_prbc_censor_event=1 AND at_risk_for_censor<>1)
            OR (deadline_censor_event=1 AND (arm<>1 OR interval_end_min<>grace_min OR at_risk_for_censor<>1))
            OR (liberal_prbc_completion=1 AND rescue_exemption_event=1)
       """)[0]==0,f"{k} grace panel rule failure")
    require(one(f"""
         SELECT count(*) FROM {k}_ccw_outcomes o
          LEFT JOIN (SELECT cohort,clone_id,sum(artificial_censor_event) events
            FROM {k}_ccw_hour GROUP BY ALL) h USING(cohort,clone_id)
          WHERE coalesce(h.events,-1)<>o.artificial_censored OR analysis_followup_hr<0
            OR icu_free_days_28d NOT BETWEEN 0 AND 28
      """)[0]==0,f"{k} clone and hour event reconciliation failed")
    print(f"{k}: cloned arms, hourly transitions, censor counts, and follow-up bounds verified.",flush=True)

def make_ccw(k):
    patient_outcomes(k)
    clone_paths(k)
    grace_panel(k)
    verify_ccw(k)
