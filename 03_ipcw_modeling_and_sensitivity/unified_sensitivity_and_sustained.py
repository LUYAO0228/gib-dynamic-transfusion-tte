"""Unified primary-specification sensitivity analyses (S1-S9) and fixed-early-weight sustained deferral."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "01_cohort_extraction"),
    str(ROOT / "02_clone_censor_weight"),
    str(ROOT / "03_ipcw_modeling_and_sensitivity"),
]

from causal_estimators import Weights, estimate, estimate_weights, weighted_quantile
from config import PROCESSED, SENSITIVITY_TABLES, SENSITIVITY_RUNTIME, SCENARIO_WEIGHTS, pq, raw
from model_features import encode_baseline, standardized
import sensitivity_common as sc_mod
import sensitivity_models as sm
from trial_data import AuthorizedRead, load_trial
from weight_model_inputs import CHRONIC
from weight_selection import mechanism_design

ERAS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]

SENS_META = {
    "S1": ("W2_6h", "Narrow Hb window 7.0 to <8.5 g/dL", None),
    "S2": ("W1_4h", "Four-hour grace window", None),
    "S3": ("W1_6h_ed2h", "ED Hb up to two hours before ICU", None),
    "S4": ("W1_6h_main", "Hourly liberal initiation model", None),
    "S5a": ("W1_6h_main", "No weight truncation", False),
    "S5b": ("W1_6h_main", "P0.5-P99.5 truncation", 0.005),
    "S5c": ("W1_6h_main", "P2.5-P97.5 truncation", 0.025),
    "S6": ("W1_6h_main", "Immediate ICU transfer censoring", None),
    "S7": ("W1_6h_main", "Include index-stay chronic ICD codes", None),
    "S8a": ("W1_6h_main", "IntakeOutput-dominant hospitals", None),
    "S8b": ("W1_6h_main", "Add infusionDrug RBC pump records", None),
    "S9-MIMIC": ("W1_6h_main", "Precisely timed in-hospital death within 28 days", None),
    "S9-eICU": ("W1_6h_main", "T0 >=24 h with APACHE IV", None),
}

_G_TRIAL = None
_G_MAIN = None
_G_COLS = None


def get_mimic_era_map():
    """Return MIMIC-IV stay_id to anchor_year_group mapping using repository-relative inputs."""
    con = sc_mod.connect()
    try:
        return (
            con.execute(
                f"""
                SELECT i.stay_id AS cohort_id, p.anchor_year_group
                FROM {raw('icu', 'icustays')} i
                JOIN {pq('MIMIC_hosp', 'patients')} p USING(subject_id)
                """
            )
            .df()
            .set_index("cohort_id")
        )
    finally:
        con.close()


def calibrate_arm0_to_arm1(t, w, m, cols_idx):
    """Apply one-sided bounded [0.7, 1.4] multiplicative calibration on Arm 0 toward Arm 1 moments."""
    if len(cols_idx) == 0:
        return 0
    x = t.cal_z[:, 1:]
    ix0 = t.landmark[0] & (m > 0)
    ix1 = t.landmark[1] & (m > 0)
    wt0 = m[ix0] * w.late[0, ix0]
    wt1 = m[ix1] * w.late[1, ix1]
    X0 = x[ix0][:, cols_idx]
    X1 = x[ix1][:, cols_idx]
    target = np.average(X1, axis=0, weights=wt1)
    avg0 = np.average(X0, axis=0, weights=wt0)
    v0 = np.average((X0 - avg0) ** 2, axis=0, weights=wt0)
    v1 = np.average((X1 - target) ** 2, axis=0, weights=wt1)
    pool = np.sqrt((v0 + v1) / 2.0)
    smd0 = np.abs(avg0 - target) / np.maximum(pool, 1e-8)
    scale = (1.0 / np.maximum(pool, 0.35)) * np.where(smd0 >= 0.095, 2.5, 1.0)

    dz0 = np.column_stack([np.ones(len(X0)), X0 - target])
    base_sw0 = (sm.stabilized(t, m)[0] * w.raw[0])[ix0]
    lo, hi = weighted_quantile(base_sw0, m[ix0], [t.rule, 1 - t.rule])
    sw0 = m[ix0] * np.clip(base_sw0, lo, hi)
    count0 = m[ix0]
    count0_sum = count0.sum()

    def fac(lin):
        return np.clip(np.exp(np.clip(lin, -12.0, 12.0)), 0.7, 1.4)

    def evaluate(theta):
        lin = dz0 @ theta
        f = fac(lin)
        nw = wt0 * f
        mass = nw.sum()
        sq = np.dot(nw / count0, nw)
        ess = (mass**2 / sq) / count0_sum
        act = (lin > np.log(0.7)) & (lin < np.log(1.4))
        dnw = (nw * act)[:, None] * dz0
        dmass = dnw.sum(axis=0)
        dess = ess * (2.0 * dmass / mass - 2.0 * ((nw / count0) @ dnw) / sq)
        mean0 = np.average(X0, axis=0, weights=nw)
        jm = (X0 - mean0).T @ dnw / mass
        sw_m = np.dot(sw0, f) / count0_sum
        jsw = (sw0 * f * act) @ dz0 / count0_sum
        res = np.r_[
            (mean0 - target) * scale,
            1.5 * (mass / wt0.sum() - 1.0),
            55.0 * min(0.0, ess - 0.505),
            55.0 * min(0.0, sw_m - 0.905),
            55.0 * max(0.0, sw_m - 1.095),
            0.04 * theta[1:],
        ]
        jac = np.vstack([
            jm * scale[:, None],
            1.5 * dmass / wt0.sum(),
            55.0 * dess * (ess < 0.505),
            55.0 * jsw * (sw_m < 0.905),
            55.0 * jsw * (sw_m > 1.095),
            np.eye(len(theta))[1:] * 0.04,
        ])
        return res, jac

    sol = least_squares(
        lambda th: evaluate(th)[0],
        np.zeros(len(cols_idx) + 1),
        jac=lambda th: evaluate(th)[1],
        max_nfev=120,
    ).x
    f_all = fac(sol[0] + (x[:, cols_idx] - target) @ sol[1:])
    w.late[0] *= f_all
    return int(((f_all[ix0] == 0.7) | (f_all[ix0] == 1.4)).any())


def _fix_arm1_weights(w, w_main):
    w.late[1] = w_main.late[1].copy()
    w.early[1, :, :6] = w_main.early[1, :, :6].copy()
    w.early[1, :, 6:] = w.late[1, :, None]
    w.raw[1] = w_main.raw[1].copy()


def build_sustained_fixed_arm1(t_sust, t_main, m, arm0_cal_cols):
    w_main = sm.build_weights(t_main, m)
    w = estimate_weights(t_sust, m)
    _fix_arm1_weights(w, w_main)
    w.tilt_bounded = w_main.tilt_bounded + calibrate_arm0_to_arm1(t_sust, w, m, arm0_cal_cols)
    return w


def _init_sustained(t_sust, t_main, cols):
    global _G_TRIAL, _G_MAIN, _G_COLS
    _G_TRIAL, _G_MAIN, _G_COLS = t_sust, t_main, cols


def _worker_sustained(seed):
    rng = np.random.default_rng(seed)
    m = np.bincount(rng.integers(_G_TRIAL.cluster_frame, size=_G_TRIAL.cluster_frame), minlength=_G_TRIAL.cluster_frame)[_G_TRIAL.clusters]
    try:
        w = build_sustained_fixed_arm1(_G_TRIAL, _G_MAIN, m, _G_COLS)
        return estimate(_G_TRIAL, m, w=w, extended=False)
    except (RuntimeError, ValueError, FloatingPointError, ConvergenceWarning) as exc:
        return {"invalid": str(exc)}


def run_sustained(workers=4, draws=500):
    """Run fixed-early-weight 24-h and 48-h sustained deferral analyses."""
    rows = []
    for db in ("mimic", "eicu"):
        t_main, w_main_locked, _ = sm.unblind(db, "Main")
        for sc in ("S_24h", "S_48h"):
            b, t_design, z, cuts = sm.design_for(db, sc)
            cal_matrix = z.to_numpy(float)
            cal_scale = np.maximum(cal_matrix.std(axis=0), 1e-8)
            t_design.cal_z = np.column_stack([np.ones(t_design.n), np.clip((cal_matrix - cal_matrix.mean(axis=0)) / cal_scale, -8, 8)])
            m_nom = np.ones(t_design.n)
            chosen = None
            candidates = []
            c_grid = (1, 2, 5, 20, 100) if db == "mimic" else (1, 2, 5, 20)
            for c in c_grid:
                t_design.c, t_design.rule = c, 0.01
                w = estimate_weights(t_design, m_nom)
                _fix_arm1_weights(w, w_main_locked)
                a0, a1, (_, tab0) = list(sm.balance(t_design, w, z))
                arms0 = [a0[1], a1[1]]
                candidates.append((float(tab0.smd.max()), c, [], arms0))
                if tab0.smd.max() < 0.10 and arms0[0]["ess_fraction"] >= 0.50 and arms0[0]["ess"] >= 150 and 0.90 <= arms0[0]["stabilized_mean"] <= 1.10:
                    chosen = candidates[-1]
                    break
            if chosen is None:
                for c in c_grid:
                    t_design.c, t_design.rule = c, 0.01
                    w_base = estimate_weights(t_design, m_nom)
                    _fix_arm1_weights(w_base, w_main_locked)
                    _, _, (_, tab0) = list(sm.balance(t_design, w_base, z))
                    for cutoff in (0.06, 0.04, 0.02, 0.0):
                        cols = np.flatnonzero(tab0.smd.to_numpy() >= cutoff)
                        w_try = Weights(w_base.early.copy(), w_base.late.copy(), 0, w_base.raw.copy())
                        calibrate_arm0_to_arm1(t_design, w_try, m_nom, cols)
                        a0, a1, (_, tabf) = list(sm.balance(t_design, w_try, z))
                        arms = [a0[1], a1[1]]
                        item = (float(tabf.smd.max()), c, cols, arms)
                        candidates.append(item)
                        if tabf.smd.max() < 0.10 and arms[0]["ess_fraction"] >= 0.50 and arms[0]["ess"] >= 150 and 0.90 <= arms[0]["stabilized_mean"] <= 1.10:
                            chosen = item
                            break
                    if chosen is not None:
                        break
            passed = chosen is not None
            if not passed:
                chosen = min(candidates, key=lambda x: (not (x[3][0]["ess_fraction"] >= 0.50 and 0.90 <= x[3][0]["stabilized_mean"] <= 1.10), x[0]))
            max_smd, best_c, best_cols, arms = chosen
            row = {
                "database": db, "scenario": sc, "n": len(b), "gate_pass": passed,
                "max_smd": max_smd, "c": best_c, "rule": 0.01, "n_cal_arm0": len(best_cols),
                "retained_deferral": arms[0]["retained"], "ess_deferral": arms[0]["ess"],
                "ess_frac_deferral": arms[0]["ess_fraction"], "sw_mean_deferral": arms[0]["stabilized_mean"],
                "retained_early": arms[1]["retained"], "ess_early": arms[1]["ess"],
                "ess_frac_early": arms[1]["ess_fraction"],
            }
            if passed:
                src = AuthorizedRead()
                try:
                    t_real = load_trial(src, db, "Main")
                finally:
                    src.con.close()
                for k, v in vars(t_design).items():
                    setattr(t_real, k, v)
                t_real.art[0] = np.minimum(t_real.art[0], cuts.reindex(b.cohort_id).fillna(np.inf))
                t_real.label = sc
                w_pt = build_sustained_fixed_arm1(t_real, t_main, np.ones(t_real.n), best_cols)
                pt = estimate(t_real, np.ones(t_real.n), w=w_pt, extended=False)
                seeds = np.random.SeedSequence([20261126, int(db == "eicu"), sum(map(ord, sc))]).generate_state(draws, dtype=np.uint64)
                with ProcessPoolExecutor(max_workers=workers, initializer=_init_sustained, initargs=(t_real, t_main, best_cols)) as pool:
                    rep = list(pool.map(_worker_sustained, seeds, chunksize=5))
                valid = [r for r in rep if "invalid" not in r]
                sc_mod.dump({"point": pt, "rep": valid, "requested": draws, "n": t_real.n, "frame": t_real.cluster_frame}, SENSITIVITY_RUNTIME / f"unified_{db}_{sc}.pkl")
                c0, c1, rd, rr, rm = [sc_mod.tail([q["metrics"][i] * s for q in valid], n0) for i, s, n0 in ((0, 100, 0.0), (1, 100, 0.0), (2, 1.0, 0.0), (3, 1.0, 1.0), (4, 1.0, 0.0))]
                row.update({
                    "events_deferral": int(pt["events"][0]), "events_early": int(pt["events"][1]),
                    "deferral_pct": 100 * pt["metrics"][0], "deferral_l95": c0["lo"], "deferral_u95": c0["hi"],
                    "early_pct": 100 * pt["metrics"][1], "early_l95": c1["lo"], "early_u95": c1["hi"],
                    "rd_pp": pt["metrics"][2], "rd_l95": rd["lo"], "rd_u95": rd["hi"], "rd_p": rd["p_empirical_plus1"],
                    "rr": pt["metrics"][3], "rr_l95": rr["lo"], "rr_u95": rr["hi"], "rr_p": rr["p_empirical_plus1"],
                    "rmst_diff": pt["metrics"][4], "rmst_l95": rm["lo"], "rmst_u95": rm["hi"], "rmst_p": rm["p_empirical_plus1"],
                    "boot_valid": len(valid), "boot_requested": draws, "clusters_resampled": t_real.cluster_frame,
                })
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(SENSITIVITY_TABLES / "unified_sustained_24h_48h.csv", index=False)
    return out


def prep_sens_trial(src, db, label, era_df):
    cohort, desc, fixed_rule = SENS_META[label]
    tt = load_trial(src, db, label, cohort, fixed_rule)
    bb = pd.read_parquet(PROCESSED / f"{db}_baseline_{cohort}.parquet").sort_values("cohort_id").reset_index(drop=True)
    if label == "S8a":
        bb = bb.loc[bb.hosp_io_dominant_ge80].reset_index(drop=True)
    elif label == "S9-eICU":
        bb = bb.loc[bb.t0_ge_24h_flag & bb.apache_iv_do_not_adjust_main.notna()].reset_index(drop=True)
    elif label == "S7":
        for k in CHRONIC:
            bb[k] = bb[k].fillna(False) | bb[k + "_index_icd_do_not_adjust_main"].fillna(False)
    tt.tilt = None
    f = encode_baseline(bb, db)
    d = mechanism_design(f) if db == "mimic" else f.model.copy()
    if db == "mimic":
        bb["anchor_year_group"] = pd.Categorical(era_df.loc[bb["cohort_id"], "anchor_year_group"], categories=ERAS)
        dummy = pd.get_dummies(bb["anchor_year_group"], prefix="calendar", drop_first=True, dtype=float).to_numpy()
        tt.x = np.column_stack([tt.x, dummy])
        tt.x0 = np.column_stack([tt.x0, dummy[tt.i0]])
        if tt.xlib is not None:
            pi, kh, dsg, tgt, xt = tt.xlib
            dsg = np.column_stack([dsg, dummy[pi]])
            if xt is not None:
                xt = np.column_stack([xt, np.repeat(dummy, tt.g, axis=0)])
            tt.xlib = (pi, kh, dsg, tgt, xt)
        f.category["anchor_year_group"] = bb["anchor_year_group"]
    if label == "S9-eICU":
        d["apache_iv_s9"] = standardized(bb.apache_iv_do_not_adjust_main.astype(float))
    diag = pd.concat([
        f.values,
        np.log1p(f.values[["creatinine", "bun_cr_ratio", "wbc", "platelets"]]).add_prefix("log1p:"),
        d[[name for name in d if ":ns" in name]],
        pd.get_dummies(f.category, prefix_sep=" = ", dtype=float),
    ], axis=1)
    if db == "eicu":
        tt.cluster_frame = int(len(np.unique(tt.clusters)))
    return tt, bb, diag, desc, fixed_rule


def _init_sens(t):
    global _G_TRIAL
    _G_TRIAL = t


def _worker_sens(seed):
    rng = np.random.default_rng(seed)
    m = np.bincount(rng.integers(_G_TRIAL.cluster_frame, size=_G_TRIAL.cluster_frame), minlength=_G_TRIAL.cluster_frame)[_G_TRIAL.clusters]
    try:
        w = sm.build_weights(_G_TRIAL, m)
        return estimate(_G_TRIAL, m, w=w, extended=False)
    except (RuntimeError, ValueError, FloatingPointError, ConvergenceWarning) as exc:
        return {"invalid": str(exc)}


def run_sensitivity_unified(workers=4, draws=500):
    """Run all S1-S9 sensitivity analyses under the unified primary specifications."""
    era_df = get_mimic_era_map()
    scenarios = {
        "mimic": ["S1", "S2", "S3", "S4", "S5a", "S5b", "S5c", "S6", "S7", "S9-MIMIC"],
        "eicu": ["S1", "S2", "S3", "S4", "S5a", "S5b", "S5c", "S6", "S8a", "S8b", "S9-eICU"],
    }
    rows = []
    src = AuthorizedRead()
    try:
        for db, labels in scenarios.items():
            for label in labels:
                if label == "S9-MIMIC":
                    tt, _, _ = sm.unblind("mimic", "Main")
                    t_s9 = load_trial(src, "mimic", "S9-MIMIC", "W1_6h_main", None)
                    tt.primary_event, tt.primary_time, tt.label = t_s9.primary_event, t_s9.primary_time, "S9-MIMIC"
                    sig = json.loads((SCENARIO_WEIGHTS / "mimic_Main.json").read_text())
                    max_smd, arms = sig["max_smd"], sig["arms"]
                    best_c, best_rule, cols, cm, passed, desc = tt.c, tt.rule, tt.cal_cols, tt.common_cal, True, SENS_META[label][1]
                else:
                    tt, _, diag, desc, locked_rule = prep_sens_trial(src, db, label, era_df)
                    cal_matrix = diag.to_numpy(float)
                    cal_scale = np.maximum(cal_matrix.std(axis=0), 1e-8)
                    tt.cal_z = np.column_stack([np.ones(tt.n), np.clip((cal_matrix - cal_matrix.mean(axis=0)) / cal_scale, -8, 8)])
                    m_nom = np.ones(tt.n)
                    if label == "S9-eICU":
                        tt.c, tt.rule, tt.cal_cols, tt.common_cal = 2.0, 0.01, [], False
                        w = sm.build_weights(tt, m_nom)
                        a0, a1, (_, tab) = list(sm.balance(tt, w, diag))
                        arms, max_smd = [a0[1], a1[1]], float(tab.smd.max())
                        best_c, best_rule, cols, cm, passed = 2.0, 0.01, [], False, bool(max_smd < 0.10)
                    else:
                        c_values = (1, 2, 5, 20, 100) if db == "mimic" else (2, 5, 20, 100, 1)
                        rule = locked_rule if locked_rule is not None else 0.01
                        chosen, candidates = None, []
                        for c in c_values:
                            tt.c, tt.rule, tt.cal_cols, tt.common_cal = c, rule, [], False
                            w = sm.build_weights(tt, m_nom)
                            a0, a1, (_, tab) = list(sm.balance(tt, w, diag))
                            arms_c = [a0[1], a1[1]]
                            item = (float(tab.smd.max()), c, rule, arms_c, [], False)
                            candidates.append(item)
                            if tab.smd.max() < 0.10 and all(a["ess_fraction"] >= 0.50 and a["ess"] >= 30 and 0.90 <= a["stabilized_mean"] <= 1.10 for a in arms_c):
                                chosen = item
                                break
                        if chosen is None:
                            ranked = sorted(candidates, key=lambda x: (not all(a["ess_fraction"] >= 0.45 and 0.85 <= a["stabilized_mean"] <= 1.15 for a in x[3]), x[0]))
                            for _, c, rule_r, _, _, _ in ranked[:4]:
                                tt.c, tt.rule, tt.cal_cols, tt.common_cal = c, rule_r, [], False
                                w0 = sm.build_weights(tt, m_nom)
                                _, _, (_, tab0) = list(sm.balance(tt, w0, diag))
                                for cutoff, common in ((0.085, False), (0.065, True), (0.04, True), (0.02, True), (0.0, True)):
                                    fails = np.flatnonzero(tab0.smd.to_numpy() >= cutoff)
                                    tt.c, tt.rule, tt.cal_cols, tt.common_cal = c, rule_r, [0, *list(fails + 1)], common
                                    w = sm.build_weights(tt, m_nom)
                                    a0, a1, (_, tabf) = list(sm.balance(tt, w, diag))
                                    arms_c = [a0[1], a1[1]]
                                    item = (float(tabf.smd.max()), c, rule_r, arms_c, tt.cal_cols, common)
                                    if tabf.smd.max() < 0.10 and all(a["ess_fraction"] >= 0.50 and a["ess"] >= 30 and 0.90 <= a["stabilized_mean"] <= 1.10 for a in arms_c):
                                        chosen = item
                                        break
                                if chosen is not None:
                                    break
                        max_smd, best_c, best_rule, arms, cols, cm = chosen
                        tt.c, tt.rule, tt.cal_cols, tt.common_cal = best_c, best_rule, cols, cm
                        passed = True

                w_pt = sm.build_weights(tt, np.ones(tt.n))
                pt = estimate(tt, np.ones(tt.n), w=w_pt, extended=False)
                seeds = np.random.SeedSequence([20261127, int(db == "eicu"), sum(map(ord, label))]).generate_state(draws, dtype=np.uint64)
                with ProcessPoolExecutor(max_workers=workers, initializer=_init_sens, initargs=(tt,)) as pool:
                    rep = list(pool.map(_worker_sens, seeds, chunksize=5))
                valid = [r for r in rep if "invalid" not in r]
                sc_mod.dump({"point": pt, "rep": valid, "requested": draws, "n": tt.n, "frame": tt.cluster_frame}, SENSITIVITY_RUNTIME / f"unified_{db}_{label}.pkl")
                c0, c1, rd, rr, rm = [sc_mod.tail([q["metrics"][i] * s for q in valid], n0) for i, s, n0 in ((0, 100, 0.0), (1, 100, 0.0), (2, 1.0, 0.0), (3, 1.0, 1.0), (4, 1.0, 0.0))]
                rows.append({
                    "database": db, "code": label, "description": desc, "n": tt.n,
                    "events_deferral": sc_mod.mask(int(pt["events"][0])), "events_early": sc_mod.mask(int(pt["events"][1])),
                    "gate_pass": passed, "max_smd": max_smd,
                    "ess_deferral": arms[0]["ess"], "ess_early": arms[1]["ess"],
                    "rule": "None" if best_rule is False else str(best_rule), "c": best_c,
                    "n_calibrated": max(0, len(cols) - 1),
                    "cal_target": "None" if len(cols) == 0 else ("common weighted moments" if cm else "full baseline moments"),
                    "deferral_pct": 100 * pt["metrics"][0], "deferral_l95": c0["lo"], "deferral_u95": c0["hi"],
                    "early_pct": 100 * pt["metrics"][1], "early_l95": c1["lo"], "early_u95": c1["hi"],
                    "rd_pp": pt["metrics"][2], "rd_l95": rd["lo"], "rd_u95": rd["hi"], "rd_p": rd["p_empirical_plus1"],
                    "rr": pt["metrics"][3], "rr_l95": rr["lo"], "rr_u95": rr["hi"], "rr_p": rr["p_empirical_plus1"],
                    "rmst_diff": pt["metrics"][4], "rmst_l95": rm["lo"], "rmst_u95": rm["hi"], "rmst_p": rm["p_empirical_plus1"],
                    "boot_valid": len(valid), "boot_requested": draws, "clusters_resampled": tt.cluster_frame,
                })
    finally:
        src.con.close()
    out = pd.DataFrame(rows)
    out.to_csv(SENSITIVITY_TABLES / "unified_sensitivity_s1_to_s9.csv", index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    SENSITIVITY_RUNTIME.mkdir(parents=True, exist_ok=True)
    SENSITIVITY_TABLES.mkdir(parents=True, exist_ok=True)
    run_sustained(workers=args.workers, draws=args.draws)
    run_sensitivity_unified(workers=args.workers, draws=args.draws)


if __name__ == "__main__":
    main()
