from cohort_database import *
from config import PRIMARY_T0_HOURS

def cohorts(k):
    # Adjacent repeats within 15 min form an episode. Do not merge across ICU entry,
    # a pRBC start, or Hb <7 stratum, which could erase strategy-defining changes.
    build(f"{k}_hb",f"""
        WITH partitions AS (
          SELECT h.*,b.entry_cross,
            (SELECT max(t) FROM {k}_rbc x WHERE x.id=h.id AND x.t<=h.t) last_rbc
          FROM {k}_hb_raw h JOIN
            (SELECT id,0 entry_cross,history_start,discharge_min FROM {k}_adapt) b USING(id)
          WHERE t BETWEEN history_start AND discharge_min
        ), prev AS (
          SELECT *,lag(t) OVER(PARTITION BY id,t<entry_cross,last_rbc
                ORDER BY t,priority,tie,hb) prev_t,
             lag(hb<7) OVER(PARTITION BY id,t<entry_cross,last_rbc
                ORDER BY t,priority,tie,hb) prev_low
          FROM partitions
        ), clusters AS (
          SELECT *,sum(CASE WHEN prev_t IS NULL OR t-prev_t>15 OR prev_low<>(hb<7)
                           THEN 1 ELSE 0 END) OVER(
             PARTITION BY id,t<entry_cross,last_rbc ORDER BY t,priority,tie,hb
             ROWS UNBOUNDED PRECEDING) episode
          FROM prev
        )
        SELECT id,t,hb,hb_source FROM clusters
        QUALIFY row_number() OVER (PARTITION BY id,t<entry_cross,last_rbc,episode
           ORDER BY priority,t,tie,hb)=1
    """)
    # Keep the candidate threshold, including for subjects failing later filters.
    # In a stay with no gray-zone Hb, measure shock through min(ICU exit,72 h).
    ed_restriction = "AND h.t>=coalesce(b.ed_min,0)" if k=="m" else ""
    build(f"{k}_candidate",f"""
        SELECT b.*,v.variant,v.ed,c.t hb_time,c.hb,c.hb_source,greatest(0,c.t) t0,
            coalesce((SELECT min(t) FROM {k}_shock x
                WHERE x.id=b.id AND x.t BETWEEN 0 AND least({PRIMARY_T0_HOURS*60},b.exit_min,coalesce(greatest(c.t,0),{PRIMARY_T0_HOURS*60}))),NULL)
                IS NOT NULL shock
        FROM {k}_adapt b CROSS JOIN
          (VALUES ('W1',9.0,false),('W2',8.5,false),('W1_ED',9.0,true)) v(variant,ceiling,ed)
        LEFT JOIN LATERAL (
          SELECT h.t,h.hb,h.hb_source
          FROM {k}_hb h WHERE h.id=b.id AND h.hb>=7 AND h.hb<v.ceiling
            AND (h.t BETWEEN 0 AND {PRIMARY_T0_HOURS*60} OR
                 (v.ed AND b.ed_link AND h.t>=-120 AND h.t<0 {ed_restriction}
                  AND NOT EXISTS(SELECT 1 FROM {k}_rbc x WHERE x.id=b.id AND x.t>=h.t AND x.t<0)))
          ORDER BY h.t,h.hb LIMIT 1
        ) c ON true
    """)
    build(f"{k}_eligible",f"""
        SELECT * FROM {k}_candidate a
        WHERE hb_time IS NOT NULL AND t0<exit_min
          AND NOT EXISTS (SELECT 1 FROM {k}_hb_raw h
                          WHERE h.id=a.id AND h.hb<7 AND h.t>=a.history_start AND h.t<a.hb_time)
          AND NOT EXISTS(SELECT 1 FROM {k}_rbc x
                         WHERE x.id=a.id AND x.t>=a.t0-360 AND x.t<a.t0
                         HAVING sum(x.ml)>=1200 OR sum(x.units)>=4)
    """)
    build(f"{k}_high_risk_eligible",f"SELECT * FROM {k}_eligible WHERE ihd OR shock")
    filter_hospital = "SEMI JOIN e_hospital_log USING(hospital)" if k=="e" else ""
    build(f"{k}_final",f"SELECT * FROM {k}_high_risk_eligible {filter_hospital}")
    assert one(f"SELECT count(*)-count(DISTINCT (id,variant)) FROM {k}_final")[0] == 0

def funnel(k):
    entries=[(label,n(f"{k}{idx}")) for idx,label in enumerate([
      "ICU stays","Acute GIB","Adults and first qualifying hospitalization and ICU stay","ICU stay of at least 6 h without major trauma or extracorporeal circulation"
    ])]
    entries.extend([
       ("High-risk candidates with ischemic heart disease or shock",one(f"SELECT count(*) FROM {k}_candidate WHERE variant='W1' AND (ihd OR shock)")[0]),
       ("Eligible high-risk stays after T0 exclusions",one(f"SELECT count(*) FROM {k}_high_risk_eligible WHERE variant='W1'")[0]),
    ])
    if k=="e":
        entries.append(("Stays in active transfusion-recording hospitals",one("SELECT count(*) FROM e_final WHERE variant='W1'")[0]))
    assert all(prev[1]>=nxt[1] for prev,nxt in zip(entries,entries[1:]))
    return tuple(count for _, count in entries)

from mimic import start_mimic, finish_mimic
from eicu import start_eicu, finish_eicu
from clinical_sources import verify_release

COHORTS = {"W1_6h_main":("W1",6), "W1_4h":("W1",4), "W2_6h":("W2",6),
           "W1_6h_ed2h":("W1_ED",6)}
EXPECTED = {"m": (1820,457,1330,13,33,1481,1797), "e": (2909,617,2194,23,98,2436,2824)}
REFERENCE_FLOW = {"m":(94458,8849,6338,6067,3818,1820), "e":(200859,10627,8976,8708,5242,2958,2909)}

def setup(k):
    if k=="m":
        verify_release()
        start_mimic()
        finish_mimic()
    else:
        start_eicu()
        finish_eicu()
    cohorts(k)
    minute_shift = "extract(second from b.entry_at)/60.0" if k=="m" else "0.0"
    build(f"{k}_trial_stays", f"""
        WITH selected AS (
           SELECT b.*, v.cohort,v.g*60 grace,{minute_shift} minute_shift,
              (SELECT min(t) FROM {k}_rbc x WHERE x.id=b.id
                     AND x.t BETWEEN b.t0 AND b.t0+60*v.g AND x.t<=b.exit_min) tx_at
           FROM {k}_final b JOIN (VALUES
             {','.join(f'({lit(name)},{lit(variant)},{hours})' for name,(variant,hours) in COHORTS.items())}
           ) v(cohort,variant,g) USING(variant)
        ), timings AS (
          SELECT *,
             (SELECT min(t) FROM {k}_hb_raw h WHERE h.id=selected.id AND h.hb<7
                         AND h.t>selected.t0 AND h.t<=least(selected.t0+selected.grace,selected.exit_min)) low_at,
             tx_at IS NOT NULL AND EXISTS(SELECT 1 FROM {k}_hb_raw h WHERE h.id=selected.id AND h.hb<7
                  AND floor(round(h.t+selected.minute_shift,8))=floor(round(selected.tx_at+selected.minute_shift,8))) tie_hb7_tx_flag
          FROM selected
        )
        SELECT *,coalesce(low_at<tx_at,false) obs_rescue,
            CASE WHEN tx_at IS NOT NULL AND (low_at IS NULL OR low_at>=tx_at) THEN 'early'
                 WHEN tx_at IS NULL AND exit_min<=t0+grace THEN 'early_exit'
                 ELSE 'restrictive' END observed_strategy
        FROM timings
    """)
    actual=one(f"""SELECT count(*) FILTER(WHERE cohort='W1_6h_main'),
         count(*) FILTER(WHERE cohort='W1_6h_main' AND observed_strategy='early'),
         count(*) FILTER(WHERE cohort='W1_6h_main' AND observed_strategy='restrictive'),
         count(*) FILTER(WHERE cohort='W1_6h_main' AND obs_rescue),
         count(*) FILTER(WHERE cohort='W1_6h_main' AND observed_strategy='early_exit'),
         count(*) FILTER(WHERE cohort='W2_6h'),count(*) FILTER(WHERE cohort='W1_6h_ed2h') FROM {k}_trial_stays""")
    flow_values=funnel(k)
    delta=tuple(b-a for a,b in zip(REFERENCE_FLOW[k],flow_values))
    require(delta==(0,)*len(delta) and actual==EXPECTED[k],
      f"{k} cohort validation failed. Selection counts expected/actual: {REFERENCE_FLOW[k]}/{flow_values}. "
      f"N/early/restrictive/rescue/exit/W2/ED expected/actual: {tuple(map(mask,EXPECTED[k]))}/{tuple(map(mask,actual))}.")
    require(n(f"{k}_trial_stays")==actual[0]*2+actual[5]+actual[6], f"{k} W1/4h count mismatch")
    if k=="e":
        require(n("e_hospital_log")==160,"eICU active hospital count mismatch")
        require(one("SELECT count(DISTINCT term) FROM e_treatment_rbc")[0]==31,"eICU RBC path count mismatch")
    print(f"{k}: selection counts and observed strategy counts verified.",flush=True)
