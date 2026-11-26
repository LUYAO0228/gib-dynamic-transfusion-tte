"""Synthetic checks for grace-window transitions, independent of the clinical databases."""
from cohort_database import *
from clone_censor import assign_policy_paths

def verify_policy_examples():
    build("qc_trial_stays","""
      SELECT * FROM (VALUES
          (1,100.0,360,NULL,900.0,100.0,NULL,0.00),
          (2,100.0,360,NULL,900.0,170.1,169.8,0.30),
          (3,100.0,360,NULL,900.0,170.9,170.1,0.30),
          (4,100.0,360,200.0,200.0,NULL,NULL,0.00),
          (5,100.0,360,NULL,180.0,NULL,NULL,0.00),
          (6,100.0,360,NULL,900.0,NULL,NULL,0.00),
          (7,100.0,360,NULL,900.0,460.0,NULL,0.00),
          (8,100.0,360,195.0,180.0,NULL,NULL,0.00)
      ) v(id,t0,grace,death_min,exit_min,tx_at,low_at,minute_shift)
    """)
    assign_policy_paths("qc")
    records = q("""SELECT id,arm,cast(artificial_rel AS DOUBLE), cast(compliance_rel AS DOUBLE),
                    cast(risk_stop_rel AS DOUBLE),censor_reason FROM qc_ccw_path ORDER BY id,arm""")
    actual = [(i,a,*[round(v,3) if v is not None else None for v in (cut,met,risk)],why)
              for i,a,cut,met,risk,why in records]
    expected = [
        (1,0,0,None,0,"early_prbc"),(1,1,None,0,0,"none"),
        (2,0,70.1,None,70.1,"early_prbc"),(2,1,None,70.1,70.1,"none"),
        (3,0,None,70.1,70.1,"none"),(3,1,None,70.9,70.9,"none"),
        (4,0,None,None,100,"none"),(4,1,None,None,100,"none"),
        (5,0,None,None,80,"none"),(5,1,80,None,80,"live_icu_exit"),
        (6,0,None,None,360,"none"),(6,1,360,None,360,"missed_deadline"),
        (7,0,360,None,360,"early_prbc"),(7,1,None,360,360,"none"),
        (8,0,None,None,80,"none"),(8,1,80,None,80,"live_icu_exit"),
    ]
    require(actual==expected,"Synthetic CCW boundary check failed")
    for t in ("qc_trial_stays","qc_ccw_path"):
        c.execute(f"DROP TABLE {t}")

if __name__ == "__main__":
    verify_policy_examples()
    print("Synthetic CCW boundary cases passed.")
    c.close()
