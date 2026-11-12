"""End-time check of the Chapter 4 comparison on published apple-drying curves (Section 4.8.8, Table 4.8).

M1 (calibrated Fickian series), M1a (Fickian series with the book descriptors PD-080/PD-081) and M5 (parallel hybrid,
M1 + GPR residual) of benchmark_ch04_physical_drying.py use the average half-thickness and normalise time with the end
time of the TEST curve, a quantity the learned models do not receive and that is known only after drying. This script
re-evaluates the three models on the same splits with the test end time replaced by
  original     the test curve's own end time (reproduces Table 4.8 / Ch04_Physical_Drying_Folds.csv, nominal_seed0),
  train_mean   the mean end time of the training curves (the variant reported in Section 4.8.8),
  nominal_600  the nominal 600 min run length,
  grid_520     the end of the common 10-520 min evaluation grid (worst case).
Predictions use the substituted end time; the targets are the unchanged published curves.

The benchmark script is loaded read-only and its 'import torch' line is skipped (PyTorch is needed only for M7, which
is not evaluated here). Scope: ten curves from one laboratory and one product; a sensitivity check of an input of the
executed comparison, not a validation.

Run from this folder:
  python check_ch04_physical_endtime.py [benchmark_ch04_physical_drying.py ../data/external/apple_drying_curves_Processes2026.csv Ch04_Physical_Drying_EndTime_Check.csv]
Output (next to this script): Ch04_Physical_Drying_EndTime_Check.csv
  columns test_end_time_used, split, model, rmse_moisture_ratio, inadmissible_curves_pct (per split; the group means are
  printed). Added on 11 November 2026 (Chapter 4 audit, item F1).
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
args = sys.argv[1:] or [str(HERE / "benchmark_ch04_physical_drying.py"),
                        str(HERE.parent / "data" / "external" / "apple_drying_curves_Processes2026.csv"),
                        str(HERE / "Ch04_Physical_Drying_EndTime_Check.csv")]
src = Path(args[0]).read_text(encoding="utf-8").replace("import torch  #", "#import torch  #")
ns = {"__name__": "endtime_check", "__file__": args[0]}
exec(compile(src, "benchmark_ch04_physical_drying", "exec"), ns)
ns["DATA"] = Path(args[1])
curves = ns["load_curves"]()
sp = ns["splits"](curves)
GROUPS = {"replicate_transfer": ("replicate_I_to_II", "replicate_II_to_I"),
          "temperature_extrapolation": ("extrapolate_80C", "extrapolate_40C"),
          "temperature_interpolation": ("interpolate_50C", "interpolate_60C", "interpolate_70C")}
MODES = ("original", "train_mean", "nominal_600", "grid_520")


def substitute(test_curves, mode, train_curves):
    out = [dict(c) for c in test_curves]
    for c in out:
        if mode == "train_mean":
            c["t_end"] = float(np.mean([x["t_end"] for x in train_curves]))
        elif mode == "nominal_600":
            c["t_end"] = 600.0
        elif mode == "grid_520":
            c["t_end"] = 520.0
    return out


res = {}
for mode in MODES:
    for name, tr, te in sp:
        mech = ns["Fick"]().fit(tr)
        book = ns["FickBook"]().fit(tr)
        hyb = ns["Hybrid"]().fit(tr, mech)
        te2 = te if mode == "original" else substitute(te, mode, tr)
        for key, m in (("M1", mech), ("M1a", book), ("M5", hyb)):
            rmse, viol = ns["evaluate"](m.predict(te2), te)
            res[(mode, name, key)] = (rmse, viol)
for mode in MODES:
    print("==", mode)
    for g, members in GROUPS.items():
        row = []
        for key in ("M1", "M1a", "M5"):
            r = np.mean([res[(mode, s, key)][0] for s in members])
            v = np.mean([res[(mode, s, key)][1] for s in members])
            row.append("%s %.5f (%.1f%%)" % (key, r, v))
        print("  %-26s" % g, " | ".join(row))
with open(args[2], "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["test_end_time_used", "split", "model", "rmse_moisture_ratio", "inadmissible_curves_pct"])
    for (mode, name, key), (r, v) in sorted(res.items()):
        w.writerow([mode, name, key, round(r, 5), round(v, 1)])
