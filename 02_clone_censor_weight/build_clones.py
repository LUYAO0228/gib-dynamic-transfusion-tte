"""Create both strategy clones, apply protocol censoring, and export risk sets."""

import cohort_database as db
from config import COHORT_STATE, PROCESSED
from eligibility import COHORTS
from check_strategy_transitions import verify_policy_examples
from clone_censor import make_ccw


def run():
    if not COHORT_STATE.is_file():
        raise FileNotFoundError("Run cohort extraction before constructing the clones.")
    verify_policy_examples()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    db.c.execute(f"ATTACH {db.lit(COHORT_STATE.as_posix())} AS cohort_inputs (READ_ONLY)")
    try:
        db.c.execute("USE cohort_inputs")
        for key, name in (("m", "mimic"), ("e", "eicu")):
            make_ccw(key)
            tables = {
                "baseline": f"{key}_baseline",
                "ccw_grace_person_hour": f"{key}_ccw_hour",
                "ccw_clone_outcomes": f"{key}_ccw_outcomes",
            }
            for cohort in COHORTS:
                for label, table in tables.items():
                    db.build("export_rows", f"SELECT * FROM {table} WHERE cohort={db.lit(cohort)}")
                    db.save("export_rows", PROCESSED / f"{name}_{label}_{cohort}.parquet")
    finally:
        db.c.execute("USE memory")
        db.c.execute("DETACH cohort_inputs")
