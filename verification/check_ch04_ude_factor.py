"""Correction-factor check of the executed UDE of Chapter 4 (Section 4.8.7, Table 4.6; cited in the text after Table 4.7).

The UDE (M6) of benchmark_ch04_drying_architectures.py multiplies the lumped first-order drying rate by (1 + g), where g
is a small neural network; the factor is not constrained to be positive (the softplus/exponential layer of Section 4.7.4
is not part of the executed UDE). For each repeat of the baseline scenario this script refits M1 and M6 exactly as the
benchmark does (same seeds, same disturbance rows) and records, per test condition (ID, OOD, CLOSURE_SHIFT): the mean
RMSE (which must equal the stored M6 value of Ch04_Executed_Drying_Comparison_Repeats.csv), the number of batches whose
factor (1 + g) becomes negative along the predicted trajectory (RK4, 0.1 h steps, 16 h), the factor at the start and at
the end of the horizon, and the mean predicted and reference final moisture.

The benchmark script is loaded read-only and its module-level 'import torch' line is skipped (M6 needs only NumPy and
SciPy). Scope: a diagnostic of the executed in-silico comparison on a declared generator; not physical validation.

Usage, run from this folder:
  python check_ch04_ude_factor.py [--workers N]
      all ten baseline repeats; writes Ch04_UDE_Correction_Factor_Check.csv next to this script (about 6 min per
      repeat on one core, UDE fitting dominates)
  python check_ch04_ude_factor.py benchmark_ch04_drying_architectures.py ../data/monte_carlo_disturbances_seed42.csv <repeat>
      one repeat; prints the record as JSON
Added on 11 November 2026 (Chapter 4 audit, items F4 and R1).
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark_ch04_drying_architectures.py"
MC = HERE.parent / "data" / "monte_carlo_disturbances_seed42.csv"
REPEATS_CSV = HERE / "Ch04_Executed_Drying_Comparison_Repeats.csv"
OUT = HERE / "Ch04_UDE_Correction_Factor_Check.csv"
CONDITIONS = ("ID", "OOD", "CLOSURE_SHIFT")


def check_repeat(bench_path, mc_path, rep):
    src = Path(bench_path).read_text(encoding="utf-8").replace("import torch  #", "#import torch  #")
    ns = {"__name__": "ude_factor_check", "__file__": str(bench_path)}
    exec(compile(src, "benchmark_ch04_drying_architectures", "exec"), ns)
    with open(mc_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cfg = ns["scenario_cfg"]("baseline")
    bud = ns["BUDGETS"][cfg["budget"]]
    n_tr, sig, srf = cfg["n_train"], cfg["sigma"], cfg["surface"]
    rng = np.random.default_rng(1000 + rep)
    pick = rng.permutation(len(rows))

    def sel(a, b):
        return [rows[i] for i in pick[a:b]]

    mk = ns["make_batches"]
    idr = {"T": (55, 70), "RH": (15, 50)}
    N = ns["N_TEST"]
    train = mk(sel(0, n_tr), idr, cfg["train_closure"], rng, sig, srf)
    tests = {"ID": mk(sel(n_tr, n_tr + N), idr, cfg["train_closure"], rng, sig, srf),
             "OOD": mk(sel(n_tr + N, n_tr + 2 * N), {"T": (72, 80), "RH": (55, 75)}, cfg["train_closure"], rng, sig, srf),
             "CLOSURE_SHIFT": mk(sel(n_tr + 2 * N, n_tr + 3 * N), idr, cfg["shift_closure"], rng, sig, srf)}
    mech = ns["Mechanistic"]().fit(train)
    ude = ns["UDE"](bud).fit(train, mech, seed=rep)
    lnk0, ea, w1, b1, w2, b2 = ns["_unpack"](ude.p)
    out = {"repeat": rep}
    for cond, tb in tests.items():
        pred = ude.predict(tb)
        pb = ns["per_batch"](pred, tb)
        # factor (1 + g) along each predicted trajectory, evaluated at the predicted states on the 0.1 h rollout grid
        T_c = np.array([b["T"] for b in tb])
        rh = np.array([b["RH"] for b in tb])
        x0 = np.array([b["X0"] for b in tb])
        T = T_c + 273.15
        xe = ns["x_eq"](rh)
        k = np.exp(lnk0) * np.exp(-ea / (ns["R_GAS"] * T))
        ts, rs = (T_c - 65.0) / 10.0, (rh - 35.0) / 20.0
        X = x0.copy()
        dt = 0.1
        fmin = np.full(len(tb), np.inf)

        def f(X):
            z = np.stack([X / x0, ts, rs], axis=1)
            g = np.tanh(z @ w1.T + b1) @ w2 + b2
            return -k * (1.0 + g) * (X - xe), 1.0 + g

        fac0 = None
        for _ in range(160):
            k1, fac = f(X)
            fmin = np.minimum(fmin, fac)
            if fac0 is None:
                fac0 = fac.copy()
            k2, _ = f(X + 0.5 * dt * k1)
            k3, _ = f(X + 0.5 * dt * k2)
            k4, _ = f(X + dt * k3)
            X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        _, facE = f(X)                                   # factor at the end of the 16 h horizon
        P = pred.reshape(len(tb), -1)
        r = np.array(pb["rmse"])
        neg = fmin < 0
        out[cond] = {"rmse_mean": float(r.mean()), "n_neg_factor": int(neg.sum()), "min_factor": float(fmin.min()),
                     "rmse_mean_neg": float(r[neg].mean()) if neg.any() else None,
                     "rmse_mean_pos": float(r[~neg].mean()) if (~neg).any() else None,
                     "rmse_max": float(r.max()), "viol_pct": 100.0 * float(np.mean(pb["viol"])),
                     "factor_start_min": float(fac0.min()), "factor_start_max": float(fac0.max()),
                     "factor_end_min": float(facE.min()), "factor_end_max": float(facE.max()),
                     "pred_final_X_mean": float(P[:, -1].mean()),
                     "true_final_X_mean": float(np.mean([b["truth"][-1] for b in tb])),
                     "x0_mean": float(x0.mean())}
    return out


def _job(rep):
    return check_repeat(BENCH, MC, rep)


def merge(records):
    import pandas as pd
    rows = [{"repeat": j["repeat"], "condition": c, **j[c]} for j in sorted(records, key=lambda x: x["repeat"])
            for c in CONDITIONS]
    df = pd.DataFrame(rows)
    ref = pd.read_csv(REPEATS_CSV)
    ref = ref[(ref.scenario == "baseline") & (ref.model.str.startswith("M6"))][["repeat", "condition", "rmse_kg_kg_db"]]
    df = df.merge(ref, on=["repeat", "condition"])
    df["abs_diff_vs_repo"] = (df.rmse_mean - df.rmse_kg_kg_db).abs()
    print(df[["repeat", "condition", "rmse_mean", "abs_diff_vs_repo", "n_neg_factor", "min_factor", "factor_end_max",
              "pred_final_X_mean", "true_final_X_mean"]].round(4).to_string())
    print("max abs diff vs stored M6 RMSE", df.abs_diff_vs_repo.max(),
          "; batches with negative factor by condition", df.groupby("condition").n_neg_factor.sum().to_dict())
    df = df.drop(columns=["rmse_kg_kg_db"]).rename(columns={"rmse_mean": "rmse_kg_kg_db",
                                                           "n_neg_factor": "batches_with_negative_factor",
                                                           "min_factor": "min_correction_factor_1_plus_g"})
    df.to_csv(OUT, index=False, lineterminator="\n")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        print(json.dumps(check_repeat(sys.argv[1], sys.argv[2], int(sys.argv[3]))))
    else:
        workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
        with ProcessPoolExecutor(max_workers=workers) as ex:
            merge(list(ex.map(_job, range(10))))
