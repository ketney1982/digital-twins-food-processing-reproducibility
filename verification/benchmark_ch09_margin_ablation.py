"""Margin ablation and policy-consistent back-off recalibration for the Chapter 9 reduced pasteuriser
(Section 9.7.8, Table 9.8).

The plant, disturbance generator, shared estimator, holding-tube loss estimator and controllers are imported
unchanged from ../run_all_benchmarks.py (ch09_declared_design, ch09_controller_setup, ch09_realisation,
ch09_simulate, ch09_conformal_backoff); nothing in that file is modified. Every configuration is run on the same
disturbance realisations as Table 9.6 (common random numbers): 200 evaluation seeds 9090000 + r and, for the
calibration studies, the 100 disjoint calibration seeds 9190000 + r.

Studies
  A  PI reference sweep. Controller 2 (PI with back-calculation anti-windup) with its tuning, initial state and
     loss estimator unchanged; only the holding-temperature reference is moved from 72.00 to 74.00 degC in steps of
     0.25 K. The archived controller uses 73.00 degC.
  B  Back-off sweep. Controller 5 (economic MPC whose critical-limit constraint is tightened by a constant back-off
     b) with b in {0, 0.34, b_arch, 1.00, 1.25, 1.50, 2.00} K. b = 0 is controller 4 and b = b_arch (the archived
     split-conformal back-off, about 0.68 K) is controller 5 of Table 9.6.
  C  Policy-consistent recalibration (audit item E06). The archived back-off is the conformal quantile of run-level
     maximum deficits observed under controller 4, i.e. under a policy WITHOUT the back-off. Here the score is
     recomputed under the policy that applies the back-off: iteration j runs controller 5 with b_j on the 100
     calibration realisations, takes the split-conformal quantile q_j (alpha = 0.05, ceil((n+1)(1-alpha))-th
     smallest run-level maximum deficit) and sets b_{j+1} = b_j + q_j. The intended stopping rule is q_j = 0 (at
     most 4 of the 100 calibration runs fall below 72 degC under the policy that is deployed); the iteration also
     stops when the increment falls below 1e-6 K (back-off stabilised) or after 8 iterations, and the CSV states which
     condition ended it. The final back-off is then evaluated once on the 200 evaluation realisations, which are
     disjoint from the calibration realisations and were not used to choose it.

Indicators per configuration: runs with at least one excursion of the true holding-tube temperature below
72.0 degC on the 0.5 s grid (count, percent, exact Clopper-Pearson 95 % interval), also restricted to the
realisations whose shared initial state is at or above 72.0 degC (in some realisations the steady state of the
nominal valve opening under that realisation's fouling, steam capacity and holding-tube loss is already below the
limit, so the run starts with an excursion that no controller or margin can prevent); the seeds with excursions;
severity (mean and maximum over runs of the largest deficit, minimum log reduction with the declared Chapter 7
D_72, z and residence time); mean time below the limit; mean holding temperature; steam energy per kg of product,
its paired relative difference against the archived PI (73.00 degC) on the same realisations and, for the economic
MPC rows, its difference from the PI sweep interpolated at the same mean holding temperature; optimisation steps
declared infeasible, solver failures and fully open valve steps. Regression anchors: PI at 73.00 degC, b = 0 and b = b_arch must reproduce controllers 2, 4 and 5 of
results/Ch09_Closed_Loop_Controller_Comparison_Benchmark.csv (rtol 1e-6).

  D  Excursion timing. For PI at 73.00 and 74.00 degC, the archived back-off, a 2.00 K back-off and the recalibrated
     back-off, every evaluation run with an excursion is re-run with a traced copy of ch09_simulate (derived from
     its source text; checked against the untraced run) to record when the true holding temperature was below the
     limit, the initial holding temperature, the inlet step and steam dip of that realisation and the number of
     fully open valve steps.

Evidence label: executed benchmark on a reduced model; not validation. The disturbance distribution, plant and
controllers are declared scenario elements; no result transfers to a physical pasteuriser or sets a margin for one.

Outputs (next to this script): Ch09_Margin_Ablation.csv (studies A-C), Ch09_Margin_Ablation_Excursion_Runs.csv (D)
Run:  python benchmark_ch09_margin_ablation.py [--workers 8] [--quick]
      --quick: 8 evaluation and 8 calibration realisations, smoke test only (no anchors, no CSV).
Run time: about 2.9 s of CPU per economic-MPC run (300 steps x ~10 ms) and 0.05 s per PI run; the full study
(about 2,000 economic-MPC runs) takes roughly 15 min on 8 worker processes of a 16-core PC.
"""
import argparse
import contextlib
import io
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

OUT_CSV = HERE / "Ch09_Margin_Ablation.csv"
OUT_RUNS_CSV = HERE / "Ch09_Margin_Ablation_Excursion_Runs.csv"
ARCHIVED_CSV = ROOT / "results" / "Ch09_Closed_Loop_Controller_Comparison_Benchmark.csv"
PI_REFS = [72.00, 72.25, 72.50, 72.75, 73.00, 73.25, 73.50, 73.75, 74.00]
BACKOFFS = [0.0, 0.34, "arch", 1.00, 1.25, 1.50, 2.00]
MAX_ITER = 8
STAB_TOL = 1.0e-6          # K; the iteration stops when the conformal increment falls below this value
LABEL = "executed_benchmark_reduced_model_not_validation"

_G = {}


def _init():
    import run_all_benchmarks as rb
    with contextlib.redirect_stdout(io.StringIO()):
        params = rb.load_parameters()
    d = rb.ch09_declared_design(params)
    _G.update(rb=rb, d=d, s=rb.ch09_controller_setup(d))


def _task(args):
    """args = (controller index, seed, back-off K, reference degC or None) -> scalar indicators of one run."""
    ci, seed, backoff, ref = args
    if not _G:
        _init()
    rb, d, s = _G["rb"], _G["d"], _G["s"]
    real = rb.ch09_realisation(d, seed)
    d_run = d if ref is None else dict(d, T_sp=float(ref))     # only the PI/LMPC reference changes; tuning from s
    out = rb.ch09_simulate(rb.CH09_CONTROLLERS[ci], real, d_run, s, backoff=float(backoff))
    t = out.pop("_times")
    out["time_ms_mean"] = 1000.0 * float(np.mean(t))
    out["seed"] = seed
    out["Th0_C"] = initial_holding_temperature(rb, d, s, real)
    out["start_below"] = float(out["Th0_C"] < d["T_CL"])
    return out


def initial_holding_temperature(rb, d, s, real):
    """Shared initial state of ch09_simulate (steady state of the nominal valve opening under this realisation's
    fouling, steam capacity, medium capacity and inlet temperature) minus the realisation's holding-tube loss.
    Below 72.0 degC the run starts below the limit whatever the controller does."""
    Cm_true = d["C_m"] * real["Cmf"]
    A0 = rb._ch09_A(d["UA"] * real["UAf"][0], d, Cm_true)
    q0 = d["Q_max"] * real["Qf"] * float(rb._ch09_valve(s["u_nom"], d["R_valve"]))
    x0 = -np.linalg.solve(A0, np.array([(q0 + d["UA_loss"] * d["T_amb"]) / Cm_true, d["mcp"] * real["T_in"][0] / d["C_p"]]))
    return float(x0[1] - real["loss"])


def _traced_sim():
    """Copy of ch09_simulate derived from its source text; the only change is that the fine-grid true holding
    temperature is also returned. Each traced run is checked against the untraced indicators (_trace_task)."""
    import inspect
    import textwrap
    rb = _G["rb"]
    src = inspect.getsource(rb.ch09_simulate)
    old = '"_times": times,'
    assert src.count(old) == 1
    src = src.replace("def ch09_simulate(", "def ch09_simulate_traced(").replace(old, old + ' "_Th": fine_Tp - real["loss"],')
    ns = dict(rb.__dict__)
    exec(textwrap.dedent(src), ns)
    return ns["ch09_simulate_traced"]


def _trace_task(args):
    """Study D: timing and disturbance context of one excursion run."""
    ci, seed, backoff, ref = args
    if not _G:
        _init()
    rb, d, s = _G["rb"], _G["d"], _G["s"]
    if "traced" not in _G:
        _G["traced"] = _traced_sim()
    real = rb.ch09_realisation(d, seed)
    d_run = d if ref is None else dict(d, T_sp=float(ref))
    out = _G["traced"](rb.CH09_CONTROLLERS[ci], real, d_run, s, backoff=float(backoff))
    chk = rb.ch09_simulate(rb.CH09_CONTROLLERS[ci], real, d_run, s, backoff=float(backoff))
    assert out["cl_max_K"] == chk["cl_max_K"] and out["energy_kJ_per_kg"] == chk["energy_kJ_per_kg"]
    h = d["dt"] / d["n_sub"]
    idx = np.where(out["_Th"] < d["T_CL"])[0]
    return {"seed": seed, "realisation_index": seed - d["seed_eval"],
            "initial_holding_temperature_C": round(initial_holding_temperature(rb, d, s, real), 4),
            "first_time_below_limit_s": round(float((idx[0] + 1) * h), 1) if idx.size else "",
            "last_time_below_limit_s": round(float((idx[-1] + 1) * h), 1) if idx.size else "",
            "largest_deficit_K": round(float(out["cl_max_K"]), 6),
            "inlet_step_time_s": round(float(real["t_step"]), 1), "inlet_step_K": round(float(real["dT_step"]), 3),
            "steam_dip_start_s": round(float(real["t_dip"]), 1), "steam_dip_full_depth_duration_s": round(float(real["dip_dur"]), 1),
            "steam_dip_depth": round(float(real["dip_depth"]), 3),
            "valve_fully_open_steps": int(out["valve_saturated_steps"])}


def run_batch(pool, ci, seeds, backoff=0.0, ref=None):
    res = list(pool.map(_task, [(ci, sd, backoff, ref) for sd in seeds], chunksize=1))
    return pd.DataFrame(res).sort_values("seed").reset_index(drop=True)


def conformal_quantile(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    return math.inf if k > n else float(np.sort(np.asarray(scores))[k - 1])


def cp_interval(k, n):
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return 100.0 * lo, 100.0 * hi


def pi_curve(pi_rows):
    """Steam use of the PI sweep as a function of mean holding temperature (piecewise-linear interpolation)."""
    pts = sorted((float(df["mean_T_hold_C"].mean()), float(df["energy_kJ_per_kg"].mean())) for df in pi_rows)
    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return lambda T: float(np.interp(T, xs, ys)) if xs[0] <= T <= xs[-1] else float("nan")


def summarise(df, study, config, par_name, par_value, data_set, pi_ref_df=None, extra=None, curve=None):
    n = len(df)
    k = int((df["cl_any"] > 0).sum())
    lo, hi = cp_interval(k, n)
    adm = df["start_below"] == 0
    n_a, k_a = int(adm.sum()), int((df.loc[adm, "cl_any"] > 0).sum())
    lo_a, hi_a = cp_interval(k_a, n_a)
    row = {
        "study": study, "configuration": config, "parameter": par_name, "parameter_value": round(float(par_value), 6),
        "realisation_set": data_set, "n_realisations": n, "runs_with_excursion": k,
        "runs_with_excursion_pct": round(100.0 * k / n, 6), "cp95_low_pct": round(lo, 6), "cp95_high_pct": round(hi, 6),
        "runs_starting_below_limit": int(n - n_a),
        "runs_with_excursion_admissible_start": k_a, "n_admissible_start": n_a,
        "runs_with_excursion_admissible_start_pct": round(100.0 * k_a / n_a, 6),
        "cp95_low_admissible_start_pct": round(lo_a, 6), "cp95_high_admissible_start_pct": round(hi_a, 6),
        "excursion_seeds": ";".join(str(int(x)) for x in df.loc[df["cl_any"] > 0, "seed"]),
        "mean_largest_deficit_K": round(float(df["cl_max_K"].mean()), 6),
        "max_largest_deficit_K": round(float(df["cl_max_K"].max()), 6),
        "mean_time_below_limit_pct": round(float(df["cl_time_pct"].mean()), 6),
        "min_log_reduction_all_runs": round(float(df["min_log_reduction"].min()), 6),
        "mean_holding_temperature_C": round(float(df["mean_T_hold_C"].mean()), 6),
        "steam_energy_kJ_per_kg": round(float(df["energy_kJ_per_kg"].mean()), 6),
        "steam_energy_vs_PI73_pct": "", "steam_vs_PI_sweep_at_equal_mean_T_pct": "",
        "infeasible_steps_total": int(df["infeasible_steps"].sum()),
        "solver_failure_steps_total": int(df["solver_failure_steps"].sum()),
        "valve_fully_open_steps_total": int(df["valve_saturated_steps"].sum()),
        "runs_with_valve_fully_open": int((df["valve_saturated_steps"] > 0).sum()),
        "conformal_quantile_K": "", "next_backoff_K": "",
        "anchor_check": "", "evidence_label": LABEL, "notes": "",
    }
    if pi_ref_df is not None:
        row["steam_energy_vs_PI73_pct"] = round(float(np.mean(100.0 * (df["energy_kJ_per_kg"].values /
                                                                      pi_ref_df["energy_kJ_per_kg"].values - 1.0))), 6)
    if curve is not None:
        ref = curve(row["mean_holding_temperature_C"])
        if math.isfinite(ref):
            row["steam_vs_PI_sweep_at_equal_mean_T_pct"] = round(100.0 * (row["steam_energy_kJ_per_kg"] / ref - 1.0), 6)
    if extra:
        row.update(extra)
    return row


def anchor(row, archived, controller):
    """Compare a sweep row with the archived Table 9.6 row of the same controller (rtol 1e-6)."""
    a = archived[archived["controller"] == controller].set_index("metric")["estimate"]
    pairs = [("runs_with_excursion_pct", "runs_with_critical_limit_excursion"),
             ("mean_largest_deficit_K", "largest_deficit_below_critical_limit"),
             ("max_largest_deficit_K", "largest_deficit_below_critical_limit_all_runs"),
             ("min_log_reduction_all_runs", "minimum_log_reduction_all_runs"),
             ("mean_holding_temperature_C", "mean_holding_temperature"),
             ("steam_energy_kJ_per_kg", "steam_energy")]
    ok = all(math.isclose(float(row[x]), float(a[y]), rel_tol=1e-6, abs_tol=1e-6) for x, y in pairs)
    row["anchor_check"] = f"{'reproduces' if ok else 'DIFFERS FROM'} {controller} of Table 9.6"
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    _init()
    d = _G["d"]
    n_eval, n_cal = (8, 8) if args.quick else (d["n_eval"], d["n_cal"])
    ev_seeds = [d["seed_eval"] + r for r in range(n_eval)]
    cal_seeds = [d["seed_cal"] + r for r in range(n_cal)]
    alpha = d["alpha_bo"]
    archived = pd.read_csv(ARCHIVED_CSV)
    rows, anchors_ok = [], []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
        # A. PI reference sweep
        pi = {}
        for ref in PI_REFS:
            pi[ref] = run_batch(pool, 1, ev_seeds, 0.0, ref)
        base = pi[73.00]
        curve = pi_curve(list(pi.values()))
        for ref in PI_REFS:
            r = summarise(pi[ref], "A_PI_reference_sweep", "2_PI_back_calculation_anti_windup", "reference_degC", ref,
                          "evaluation", base, {"notes": "PI tuning, initial state and loss estimator unchanged; only the reference moves"})
            if ref == 73.00 and not args.quick:
                anchors_ok.append(anchor(r, archived, "2_PI_back_calculation_anti_windup"))
            rows.append(r)
        print(f"A done ({time.perf_counter() - t0:.0f} s)", flush=True)

        # C0. archived back-off: conformal quantile of controller-4 scores on the calibration realisations
        c4 = run_batch(pool, 3, cal_seeds, 0.0)
        b_arch = conformal_quantile(c4["cl_max_K"].values, alpha)
        rows.append(summarise(c4, "C_policy_consistent_recalibration", "iteration_0_controller_4_scores", "backoff_K", 0.0,
                              "calibration", None, {"conformal_quantile_K": round(b_arch, 6), "next_backoff_K": round(b_arch, 6),
                                                    "notes": "archived calibration: scores observed under the policy without back-off"}))
        print(f"archived back-off {b_arch:.6f} K ({time.perf_counter() - t0:.0f} s)", flush=True)

        # B. back-off sweep on the evaluation realisations
        bdf = {}
        for b in BACKOFFS:
            bv = b_arch if b == "arch" else float(b)
            df = run_batch(pool, 4, ev_seeds, bv)
            bdf[b] = df
            ctrl = "4_Nominal_EMPC_hard_critical_limit" if bv == 0.0 else "5_EMPC_conformal_backoff"
            r = summarise(df, "B_backoff_sweep", ctrl, "backoff_K", bv, "evaluation", base,
                          {"notes": "archived split-conformal back-off" if b == "arch" else ("no back-off" if bv == 0.0 else "")},
                          curve=curve)
            if not args.quick and (b == "arch" or bv == 0.0):
                anchors_ok.append(anchor(r, archived, ctrl))
            rows.append(r)
            print(f"B b={bv:.3f} K: {r['runs_with_excursion']}/{n_eval} runs below 72 degC ({time.perf_counter() - t0:.0f} s)", flush=True)

        # C. policy-consistent recalibration
        b = b_arch
        for j in range(1, MAX_ITER + 1):
            df = run_batch(pool, 4, cal_seeds, b)
            q = conformal_quantile(df["cl_max_K"].values, alpha)
            rows.append(summarise(df, "C_policy_consistent_recalibration", f"iteration_{j}_controller_5_scores", "backoff_K", b,
                                  "calibration", None, {"conformal_quantile_K": round(q, 6), "next_backoff_K": round(b + q, 6),
                                                        "notes": "scores observed under the policy that applies the back-off"}))
            print(f"C iter {j}: b={b:.6f} K, calibration runs below limit {int((df['cl_any'] > 0).sum())}/{n_cal}, "
                  f"q={q:.6f} K ({time.perf_counter() - t0:.0f} s)", flush=True)
            if q <= STAB_TOL:
                break
            b = b + q
        if q == 0.0:
            note = f"stopping rule met: conformal quantile zero (at most {n_cal - math.ceil((n_cal + 1) * (1 - alpha))} calibration runs below the limit)"
        elif q <= STAB_TOL:
            note = (f"back-off stabilised (increment {q:.1e} K <= {STAB_TOL:.0e} K) without meeting the stopping rule: "
                    f"{int((df['cl_any'] > 0).sum())} of {n_cal} calibration runs remain below the limit, "
                    f"{int(df['start_below'].sum())} of them starting below it")
        else:
            note = f"not stabilised after {MAX_ITER} iterations"
        print("C:", note, flush=True)
        df = run_batch(pool, 4, ev_seeds, b)
        rows.append(summarise(df, "C_policy_consistent_recalibration", "final_backoff_on_evaluation_realisations", "backoff_K", b,
                              "evaluation", base, {"notes": note + "; back-off chosen on calibration realisations only"}, curve=curve))
        print(f"C final b={b:.6f} K: {rows[-1]['runs_with_excursion']}/{n_eval} evaluation runs below limit", flush=True)

        # D. timing and disturbance context of every excursion run of selected configurations
        jobs = [("PI_73.00", 1, 0.0, 73.00, pi[73.00]), ("PI_74.00", 1, 0.0, 74.00, pi[74.00]),
                (f"EMPC_backoff_{b_arch:.6f}", 4, b_arch, None, bdf["arch"]), ("EMPC_backoff_2.000000", 4, 2.00, None, bdf[2.00]),
                (f"EMPC_recalibrated_backoff_{b:.6f}", 4, b, None, df)]
        trace_rows = []
        for label, ci, bo, ref, df_ in jobs:
            seeds = [int(x) for x in df_.loc[df_["cl_any"] > 0, "seed"]]
            for x in pool.map(_trace_task, [(ci, sd, bo, ref) for sd in seeds], chunksize=1):
                trace_rows.append(dict(configuration=label, **x))
        print(f"D done: {len(trace_rows)} excursion runs traced ({time.perf_counter() - t0:.0f} s)", flush=True)

    out = pd.DataFrame(rows)
    print(out[["study", "parameter_value", "realisation_set", "runs_with_excursion_pct", "cp95_low_pct", "cp95_high_pct",
               "mean_largest_deficit_K", "max_largest_deficit_K", "mean_holding_temperature_C", "steam_energy_kJ_per_kg",
               "steam_energy_vs_PI73_pct", "conformal_quantile_K", "anchor_check"]].to_string())
    print(f"run time {time.perf_counter() - t0:.0f} s")
    if args.quick:
        print("quick smoke test: no CSV written")
        return
    if not all(anchors_ok):
        print("WARNING: at least one regression anchor differs from Table 9.6", flush=True)
    out.to_csv(OUT_CSV, index=False, lineterminator="\n")
    tr = pd.DataFrame(trace_rows)
    tr["evidence_label"] = LABEL
    tr.to_csv(OUT_RUNS_CSV, index=False, lineterminator="\n")
    print(f"saved {OUT_CSV.name} ({len(out)} rows) and {OUT_RUNS_CSV.name} ({len(tr)} rows); "
          f"anchors reproduced: {sum(anchors_ok)}/{len(anchors_ok)}")


if __name__ == "__main__":
    main()
