import cohort_database as a
from config import MAIN_COHORT, SENSITIVITY_INPUTS
from event_features import bounds, m_events, e_events


def write_inputs(k):
    db = "mimic" if k == "m" else "eicu"
    WORK = SENSITIVITY_INPUTS
    WORK.mkdir(parents=True, exist_ok=True)
    def save_select(name,sql):
        p=(WORK/f"{db}_{name}.parquet").with_suffix(".partial.parquet")
        a.c.execute(f"COPY ({sql}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",[str(p)])
        p.replace(WORK/f"{db}_{name}.parquet")
    a.build("sel", f"SELECT * FROM {k}_trial_stays WHERE cohort={a.lit(MAIN_COHORT)}")
    ac="ami" if k=="m" else "acute"
    era="p.anchor_year_group, p.anchor_year, year(s.entry_at) AS shifted_icu_year" if k=="m" else "NULL::VARCHAR anchor_year_group, NULL::INTEGER anchor_year, NULL::INTEGER shifted_icu_year"
    erajoin=f"JOIN {a.raw('hosp','patients')} p ON p.subject_id=s.person" if k=="m" else ""
    save_select("baseline_extensions",f"""
       SELECT s.id cohort_id, {era}, s.shock, s.ihd, s.{ac} acute_acs, (s.exit_min-s.t0)/60 observation_icu_hr
       FROM sel s {erajoin}
    """)
    save_select("rbc",f"""SELECT s.id cohort_id,(r.t-s.t0)/60 hr,r.ml,r.units
                      FROM sel s JOIN {k}_rbc r ON s.id=r.id WHERE r.t BETWEEN s.t0 AND least(s.discharge_min,s.t0+10080)""")
    a.c.execute(f"CREATE OR REPLACE TEMP TABLE {k}_trial_stays AS SELECT * FROM sel")
    bounds(k)
    a.c.execute(f"UPDATE {k}_need SET feature_max=(SELECT t0+10080 FROM sel WHERE sel.id={k}_need.id)")
    (m_events if k=="m" else e_events)()
    save_select("hb",f"""
       SELECT s.id cohort_id,(z.t-s.t0)/60 hr,(z.known-s.t0)/60 known_hr,z.val hb,z.pref
       FROM sel s JOIN {k}_labe z USING(id) WHERE z.kind='hb'
       QUALIFY row_number() OVER(PARTITION BY s.id,z.t,z.known ORDER BY z.pref,z.val)=1
    """)
    save_select("vitals",f"""
       SELECT s.id cohort_id,(z.t-s.t0)/60 hr,(z.known-s.t0)/60 known_hr,z.kind,z.val,z.pref
       FROM sel s JOIN {k}_vitale z USING(id) WHERE z.kind IN ('sbp','map') AND z.t<=s.t0+2880
    """)
    if k=="m":
        vaso=f"""SELECT s.id cohort_id,(date_diff('second',s.entry_at,{a.ts('v.starttime')})/60-s.t0)/60 hr,
             (date_diff('second',s.entry_at,{a.ts('v.endtime')})/60-s.t0)/60 end_hr,TRUE active
             FROM sel s JOIN {a.pq('mimiciv_derived','vasoactive_agent')} v ON v.stay_id=s.id
             WHERE greatest(v.norepinephrine,v.epinephrine,v.phenylephrine,v.vasopressin,v.dopamine)>0"""
    else:
        vaso="""SELECT s.id cohort_id,(v.t-s.t0)/60 hr,(v.t-s.t0)/60 end_hr,
                greatest(v.drug_rate,v.infusion_rate)>0 active
                FROM sel s JOIN e_press v USING(id) WHERE v.t<=s.t0+2880 AND v.t>=s.t0-360"""
    save_select("vaso",vaso)
    save_select("blood",f"""SELECT s.id cohort_id,(z.t-s.t0)/60 hr,(z.known-s.t0)/60 known_hr
             FROM sel s JOIN {k}_blood z USING(id) WHERE z.t BETWEEN s.t0 AND s.t0+2880""")
