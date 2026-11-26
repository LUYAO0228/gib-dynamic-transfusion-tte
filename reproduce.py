"""Run cohort extraction, clone construction, or outcome modeling."""

import argparse
import os
from pathlib import Path
import sys

DIRECTORIES = (
    "01_cohort_extraction",
    "02_clone_censor_weight",
    "03_ipcw_modeling_and_sensitivity",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("01", "02", "03", "check-transitions"))
    parser.add_argument("--workers", type=int, default=4, help="Bootstrap workers for step 03")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    root = Path(__file__).resolve().parent
    sys.path[:0] = [str(root / name) for name in DIRECTORIES]
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

    if args.step == "03":
        from run_analysis import run

        run(args.workers)
        return

    import cohort_database

    cohort_database.initialize()
    try:
        if args.step == "01":
            from build_cohorts import run
        elif args.step == "02":
            from build_clones import run
        else:
            from check_strategy_transitions import verify_policy_examples as run
        run()
        if args.step == "check-transitions":
            print("Synthetic strategy transition checks passed.")
    finally:
        cohort_database.c.close()


if __name__ == "__main__":
    main()
