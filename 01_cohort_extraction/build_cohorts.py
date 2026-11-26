"""Extract eligible stays and persist the inputs required for strategy cloning."""

import cohort_database as db
from config import COHORT_STATE, DUCKDB_TEMP
from eligibility import setup
from cohort_features import make_features
from sensitivity_inputs import write_inputs


def run():
    DUCKDB_TEMP.mkdir(parents=True, exist_ok=True)
    partial = COHORT_STATE.with_suffix(".partial.duckdb")
    partial.unlink(missing_ok=True)
    db.c.execute(f"ATTACH {db.lit(partial.as_posix())} AS cohort_inputs")
    try:
        for key, population in (("m", "m0"), ("e", "e0")):
            setup(key)
            make_features(key)
            tables = [population] + [
                f"{key}_{suffix}" for suffix in (
                    "trial_stays", "rbc", "baseline", "covariate_windows", "labe", "blood",
                )
            ]
            if key == "m":
                tables.append("m_labs")
            for table in tables:
                db.c.execute(f"CREATE TABLE cohort_inputs.main.{table} AS SELECT * FROM {table}")
            write_inputs(key)
    finally:
        db.c.execute("DETACH cohort_inputs")
    partial.replace(COHORT_STATE)
