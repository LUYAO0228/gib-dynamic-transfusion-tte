"""Run the reference IPCW model, calendar and timing analyses, and bias analyses."""

from config import DBS, SENSITIVITY_RUNTIME, SENSITIVITY_TABLES, SCENARIO_WEIGHTS
from fit_reference_weights import run_lock
from bootstrap_trials import run_dataset
from reference_results import collect, csv
from settings import TABLES as REFERENCE_TABLES
from sensitivity_models import blind, unblind, run_boot
from calendar_and_subgroups import main_export
from timing_and_sustained_deferral import describe_post_grace, timing_and_feasibility
from bias_and_retransfusion import export_retransfusion, qba, retransfusion

SCENARIOS = {
    "mimic": ("Main", "S_2014_2022", "S_2011_2022", "S_T0_le48h", "S_T0_le24h", "S_24h", "S_48h"),
    "eicu": ("Main", "S_T0_le48h", "S_T0_le24h", "S_24h", "S_48h", "eicu_intake_output_dominant"),
}


def run(workers):
    for path in (SENSITIVITY_RUNTIME, SENSITIVITY_TABLES, SCENARIO_WEIGHTS):
        path.mkdir(parents=True, exist_ok=True)
    run_lock()
    for name in DBS:
        run_dataset(name, "all", workers)
    tables, diagnostics = collect()
    names = ("outcomes", "subgroups", "sensitivity", "risk_curves", "hemoglobin_effect_curve")
    for name, table in zip(names, tables):
        csv(table, REFERENCE_TABLES / f"{name}.csv")
    csv(diagnostics, REFERENCE_TABLES / "bootstrap_diagnostics.csv")

    for name in DBS:
        for scenario in SCENARIOS[name]:
            blind(name, scenario)
            fitted = unblind(name, scenario)
            if fitted is None:
                print(f"{name} {scenario}: weight criteria not met.")
                continue
            trial, weights, _ = fitted
            count = 1000 if scenario == "Main" else 300 if trial.g > 6 else 500
            run_boot(trial, weights, count, workers, scenario == "Main",
                     SENSITIVITY_RUNTIME / f"boot_{name}_{scenario}.pkl")
            if name == "eicu" and scenario == "Main":
                trial.cluster_frame = 160
                run_boot(trial, weights, 1000, workers, False,
                         SENSITIVITY_RUNTIME / "boot_eicu_Main160.pkl")
    main_export()
    describe_post_grace()
    timing_and_feasibility()
    qba()
    for name in DBS:
        retransfusion(name, workers, "Main")
    retransfusion("eicu", workers, "eicu_intake_output_dominant")
    export_retransfusion()
