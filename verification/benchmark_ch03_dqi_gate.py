"""Executed in-silico test of a data-readiness gate on generated telemetry with labelled faults (Section 3.9.3).

The assigned diagnostic vectors of results/Ch03_Data_Readiness_Gate_ROC_Benchmark.csv (including the constructed
score 0.996) are NOT read, changed or re-estimated here. This script runs the evaluation protocol of Section 3.9.3
(Table 3.6) once, on generated data, so that its reporting fields come from an executed experiment.

Generated telemetry (assigned generator; 1 Hz; production batches of 2 h; UHT holding-tube outlet near 138 degC):
  heating-medium temperature S = set-point + offset (2-4 K) + AR(1) disturbance (tau 600 s, SD 0.25 K)
      - genuine process excursions (Poisson 0.8 per batch; raised-cosine dips of 1.5-6 K lasting 120-480 s);
  true product temperature T = first-order lag of (S - offset), tau 15 s;
  two redundant product probes A and B (lags 2 s and 3 s, white noise SD 0.10 K, calibration offset SD 0.05 K,
      resolution 0.01 K) and the measured heating-medium channel (noise SD 0.10 K);
  labelled sensor faults: spikes on probe A or B (Poisson 1.5 per batch, 1-3 samples, |amplitude| 0.6-4.0 K),
      frozen values from dropouts with last-value hold on A or B (probability 0.5 per batch, 60-600 s), and linear
      drift of 0.5-3.0 K/h (either sign) from onset to the end of the batch (at least 20 min), either on one probe
      (probability 0.4 per batch) or common-mode, identical on both probes (probability 0.2 per batch; a class that
      the redundant-probe residual cannot see by construction).
  The first 10 min of every batch are fault- and excursion-free (verified start-up window); events do not overlap.
  Rolling-window statistics make no decision before their window is filled (first 30 s or 60 s).

Detectors (each returns a per-channel score; a channel is rejected when its score exceeds the multiplier m):
  none          admits every sample;
  fixed         fixed limits |y_j - set-point| > m (K), per probe;
  hampel        causal Hampel identifier, trailing window of 31 samples, |y - median| / (1.4826 MAD) > m;
  discrepancy   redundant-pair limit |(y_A - y_B) - start-up offset| > m (K); rejects both probes (no isolation);
  gate          reduced implementation of Algorithm 3.1 (Stages 1, 3 and 6) for this sensor set:
                  s1 spike/step: |r - trailing median_61(r)| / (6 sigma_r), r = y_A - y_B;
                  s2 drift: EWMA (lambda 0.01) of the spike-cleaned standardised residual / (6 sigma_EWMA);
                  s3 completeness: run length of identical consecutive values / 5 samples (per probe);
                  isolation of the faulty probe by its deviation from the heating-medium model prediction;
                  physical-excursion mode (retain, do not reject) when |mean probe - set-point| > 1 K without alarm.
Splits: 40 development batches (seeds 3000-3039) fit sigma_r, the model lag and every threshold; 80 test batches
(seeds 4000-4079) are scored once. Threshold rule, fixed before testing: the smallest m on the grid with at most
0.5 false-alarm events per nominal operating hour on the development batches.
Metrics on test batches: episode sensitivity (any alarm on either probe within the episode, plus 5 s for spikes
and 30 s otherwise), detection delay, false-alarm events per nominal hour (alarms merged when < 30 s apart;
nominal = outside fault episodes + 60 s guard; genuine excursions count as nominal), false-alarm probability per
nominal sample, post-gate mean absolute error of the admitted output (mean of the admitted probes) against the true
product temperature during fault episodes, and genuine-excursion samples withheld (both probes rejected).
Intervals: percentile bootstrap over test batches (2000 resamples, seed 12345); Clopper-Pearson intervals for
sensitivities treat episodes as independent units. A threshold sweep (test operating curve) is stored as well.

Design history (disclosed): after a first run of an earlier version, the start-up handling of the rolling-window
statistics was corrected (padding with the first sample had produced spurious alarms) and the common-mode drift class
was added; the base settings of the detectors and the threshold rule were not changed. The stored CSV is the output
of this version (deterministic: identical on rerun with the same software versions).

Evidence label: executed on generated telemetry; not industrial data; not validation of the gate. The result
depends on the generator, the fault classes and the information given to each detector (only the discrepancy
limit and the gate use the redundant probe; only the gate uses the heating-medium channel).

Output: Ch03_DQI_Gate_Executed_Experiment.csv (next to this script).
Run:    python benchmark_ch03_dqi_gate.py      (about 1-2 min on a desktop PC; numpy, scipy, pandas only)
"""
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from scipy.stats import beta

OUT = Path(__file__).resolve().parent / "Ch03_DQI_Gate_Executed_Experiment.csv"
EVID = "executed_on_generated_telemetry_not_industrial_data_not_validation"

DUR, START = 7200, 600                     # s per batch, fault-free start-up window (s)
N_DEV, N_TEST, SEED_DEV, SEED_TEST = 40, 80, 3000, 4000
B_FA = 0.5                                  # allowed false-alarm events per nominal hour (development batches)
GUARD, MERGE = 60, 30                       # s
GRACE = {"spike": 5, "frozen": 30, "drift": 30, "drift_common": 30}
TYPES = ["all", "spike", "frozen", "drift", "drift_common"]
N_BOOT, SEED_BOOT = 2000, 12345
NOISE, OFFSET_SD, RES = 0.10, 0.05, 0.01
GRIDS = {
    "fixed": np.round(np.arange(0.20, 8.0001, 0.05), 3),
    "hampel": np.round(np.arange(2.0, 60.0001, 0.25), 3),
    "discrepancy": np.round(np.arange(0.05, 3.0001, 0.01), 3),
    "gate": np.round(np.arange(0.30, 5.0001, 0.02), 3),
    "none": np.array([1.0]),
}
DETECTORS = ["none", "fixed", "hampel", "discrepancy", "gate"]
LABELS = {"none": "No gate", "fixed": "Fixed limits", "hampel": "Hampel identifier",
          "discrepancy": "Redundant-pair discrepancy limit", "gate": "Proposed gate (reduced Algorithm 3.1)"}
UNITS = {"fixed": "K", "hampel": "robust SD", "discrepancy": "K", "gate": "multiple of base thresholds", "none": "-"}


# ----------------------------------------------------------------------------------------------- generator
def lag(x, tau, x0=None):
    a = math.exp(-1.0 / tau)
    y0 = x[0] if x0 is None else x0
    y, _ = lfilter([1.0 - a], [1.0, -a], x, zi=[a * y0])
    return y


def generate_batch(seed):
    rng = np.random.default_rng(seed)
    n = DUR
    t = np.arange(n, dtype=float)
    T_sp = rng.uniform(137.5, 139.5)
    d_hs = rng.uniform(2.0, 4.0)
    phi = math.exp(-1.0 / 600.0)
    w = rng.normal(0.0, 0.25 * math.sqrt(1 - phi ** 2), n)
    ar, _ = lfilter([1.0], [1.0, -phi], w, zi=[phi * rng.normal(0.0, 0.25)])
    S = T_sp + d_hs + ar
    occupied = [(0, START)]

    def place(length, lo, hi):
        for _ in range(500):
            if hi - length <= lo:
                return None
            s = int(rng.integers(lo, hi - length))
            e = s + length
            if all(e + 120 <= a or s >= b + 120 for a, b in occupied):
                occupied.append((s, e))
                return s, e
        return None

    faults, excursions = [], []
    u_drift = rng.random()
    limit = n - 60
    drift = None
    if u_drift < 0.6:
        onset = int(rng.integers(START + 1200, n - 1200))
        common = u_drift >= 0.4                    # common-mode drift: identical bias on both probes
        drift = {"type": "drift_common" if common else "drift", "channel": 2 if common else int(rng.integers(0, 2)),
                 "start": onset, "end": n - 1, "rate": rng.uniform(0.5, 3.0) * rng.choice([-1.0, 1.0])}
        limit = onset - 120
    for _ in range(rng.poisson(0.8)):
        pl = place(int(rng.integers(120, 481)), START, limit)
        if pl:
            s, e = pl
            depth = rng.uniform(1.5, 6.0)
            S[s:e] -= depth * 0.5 * (1 - np.cos(2 * np.pi * (t[s:e] - s) / (e - s)))
            excursions.append((s, min(e + 60, n - 1)))
    for _ in range(rng.poisson(1.5)):
        pl = place(int(rng.integers(1, 4)), START, limit)
        if pl:
            faults.append({"type": "spike", "channel": int(rng.integers(0, 2)), "start": pl[0], "end": pl[1] - 1,
                           "amp": rng.choice([-1.0, 1.0]) * rng.uniform(0.6, 4.0)})
    if rng.random() < 0.5:
        pl = place(int(rng.integers(60, 601)), START, limit)
        if pl:
            faults.append({"type": "frozen", "channel": int(rng.integers(0, 2)), "start": pl[0], "end": pl[1] - 1})
    if drift:
        faults.append(drift)
    T = lag(S - d_hs, 15.0, x0=T_sp + ar[0])
    y = np.vstack([lag(T, 2.0) + rng.normal(0, OFFSET_SD) + rng.normal(0, NOISE, n),
                   lag(T, 3.0) + rng.normal(0, OFFSET_SD) + rng.normal(0, NOISE, n)])
    for f in faults:
        if f["type"] == "spike":
            y[f["channel"], f["start"]:f["end"] + 1] += f["amp"]
        elif f["type"] in ("drift", "drift_common"):
            ch = [0, 1] if f["channel"] == 2 else [f["channel"]]
            y[ch, f["start"]:] += f["rate"] * (t[f["start"]:] - f["start"]) / 3600.0
    y = np.round(y / RES) * RES
    for f in faults:
        if f["type"] == "frozen":
            y[f["channel"], f["start"]:f["end"] + 1] = y[f["channel"], f["start"] - 1]
    Sm = np.round((S + rng.normal(0, NOISE, n)) / RES) * RES
    nominal = np.ones(n, bool)
    infault = np.zeros(n, bool)
    typemask = {ty: np.zeros(n, bool) for ty in TYPES[1:]}
    for f in faults:
        nominal[f["start"]:min(n, f["end"] + 1 + GUARD)] = False
        infault[f["start"]:f["end"] + 1] = True
        typemask[f["type"]][f["start"]:f["end"] + 1] = True
    excmask = np.zeros(n, bool)
    for s, e in excursions:
        excmask[s:e + 1] = True
    excmask &= ~infault
    return {"T": T, "y": y, "Sm": Sm, "T_sp": T_sp, "faults": faults, "excursions": excursions,
            "nominal": nominal, "infault": infault, "excmask": excmask, "typemask": typemask}


# ----------------------------------------------------------------------------------------------- detectors
def trailing(x, w):
    xp = np.concatenate([np.full(w - 1, x[0]), x])
    return sliding_window_view(xp, w)


def run_length_identical(x):
    same = np.concatenate([[False], np.diff(x) == 0.0])
    idx = np.arange(len(x))
    last_break = np.maximum.accumulate(np.where(~same, idx, 0))
    return idx - last_break           # number of consecutive repeats of the current value


def ewma(x, lam, z0=0.0):
    z, _ = lfilter([lam], [1.0, -(1 - lam)], x, zi=[(1 - lam) * z0])
    return z


class Fitted:
    sigma_r = None
    tau_hat = None


def model_prediction(b, tau):
    ybar = b["y"].mean(axis=0)
    d0 = np.median(b["Sm"][:START] - ybar[:START])
    return lag(b["Sm"] - d0, tau, x0=ybar[0])


def fit_on_development(dev):
    res = []
    for b in dev:
        r = b["y"][0] - b["y"][1]
        r0 = np.median(r[:START])
        res.append((r - r0)[b["nominal"]])
    res = np.concatenate(res)
    Fitted.sigma_r = 1.4826 * np.median(np.abs(res - np.median(res)))
    best = None
    for tau in np.arange(5.0, 40.01, 1.0):
        sse = 0.0
        for b in dev:
            m = b["nominal"]
            sse +=np.sum((b["y"].mean(axis=0) - model_prediction(b, tau))[m] ** 2)
        if best is None or sse < best[0]:
            best = (sse, tau)
    Fitted.tau_hat = best[1]


def scores(det, b):
    """Return (sA, sB, reject_both_flag). A channel is rejected when its score > m."""
    y = b["y"]
    n = y.shape[1]
    if det == "none":
        z = np.zeros(n)
        return z, z, False
    if det == "fixed":
        return np.abs(y[0] - b["T_sp"]), np.abs(y[1] - b["T_sp"]), False
    if det == "hampel":
        out = []
        for j in range(2):
            win = trailing(y[j], 31)
            med = np.median(win, axis=1)
            mad = 1.4826 * np.median(np.abs(win - med[:, None]), axis=1)
            s = np.abs(y[j] - med) / np.maximum(mad, RES)
            s[:30] = 0.0                      # window not yet filled: no decision
            out.append(s)
        return out[0], out[1], False
    r = y[0] - y[1]
    r0 = np.median(r[:START])
    if det == "discrepancy":
        s = np.abs(r - r0)
        return s, s, True
    # proposed gate
    sr = Fitted.sigma_r
    med_r = np.median(trailing(r, 61), axis=1)
    s1 = np.abs(r - med_r) / (6.0 * sr)
    s1[:60] = 0.0                             # window not yet filled: no decision
    clean = np.where(s1 > 1.0, med_r, r)
    lam = 0.01
    z = ewma((clean - r0) / sr, lam)
    s2 = np.abs(z) / (6.0 * math.sqrt(lam / (2 - lam)))
    s3 = np.vstack([run_length_identical(y[0]), run_length_identical(y[1])]) / 5.0
    That = model_prediction(b, Fitted.tau_hat)
    dev_inst = np.vstack([np.abs(y[j] - That - np.median((y[j] - That)[:START])) for j in range(2)])
    dev_slow = np.vstack([np.abs(ewma(y[j] - That - np.median((y[j] - That)[:START]), 0.05,
                                      z0=0.0)) for j in range(2)])
    use_slow = s2 >= s1
    iso = np.where(use_slow, np.argmax(dev_slow, axis=0), np.argmax(dev_inst, axis=0))
    s12 = np.maximum(s1, s2)
    sA = np.maximum(s3[0], np.where(iso == 0, s12, 0.0))
    sB = np.maximum(s3[1], np.where(iso == 1, s12, 0.0))
    return sA, sB, False


# ----------------------------------------------------------------------------------------------- evaluation
def alarm_events(alarm):
    idx = np.flatnonzero(alarm)
    if idx.size == 0:
        return idx
    starts = np.concatenate([[True], np.diff(idx) > MERGE])
    return idx[starts]


def evaluate(b, sA, sB, both, m, full=True):
    pair = np.maximum(sA, sB) > m
    exc_mode = (np.abs(b["y"].mean(axis=0) - b["T_sp"]) > 1.0) & ~pair   # gate: retain as process excursion
    cs = np.concatenate([[0], np.cumsum(pair)])
    det = []
    for f in b["faults"]:
        lo = f["start"]
        hi = min(DUR - 1, f["end"] + GRACE[f["type"]])
        hit = cs[hi + 1] - cs[lo] > 0
        d = {"type": f["type"], "channel": f["channel"], "detected": bool(hit), "delay": np.nan, "iso_ok": np.nan,
             "exc_mode": bool(exc_mode[lo:hi + 1].any())}
        if hit:
            k = lo + int(np.argmax(pair[lo:hi + 1]))
            d["delay"] = float(k - lo)
            if not both:
                rej = (sA[k] > m, sB[k] > m)
                if f["channel"] == 2:
                    d["iso_ok"] = float(rej[0] and rej[1])
                else:
                    d["iso_ok"] = float(rej[f["channel"]] and not rej[1 - f["channel"]])
        det.append(d)
    ev = alarm_events(pair)
    nominal = b["nominal"]
    out = {"det": det, "fa_events": int(np.sum(nominal[ev])) if ev.size else 0,
           "fa_events_exc": int(np.sum((nominal & b["excmask"])[ev])) if ev.size else 0,
           "nominal_s": int(nominal.sum()), "nominal_alarm_s": int(np.sum(pair & nominal))}
    if not full:
        return out
    if both:
        admA = admB = ~pair
    else:
        admA, admB = ~(sA > m), ~(sB > m)
    y = b["y"]
    num = np.where(admA, y[0], 0.0) + np.where(admB, y[1], 0.0)
    cnt = admA.astype(int) + admB.astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        outp = num / cnt
    avail = cnt > 0
    err = np.abs(outp - b["T"])
    fm = b["infault"]
    out.update({"fault_s": int(fm.sum()), "fault_avail_s": int(np.sum(fm & avail)),
                "fault_abs_err": float(np.sum(err[fm & avail])),
                "nom_avail_s": int(np.sum(nominal & avail)), "nom_abs_err": float(np.sum(err[nominal & avail])),
                "exc_s": int(b["excmask"].sum()), "exc_withheld_s": int(np.sum(b["excmask"] & ~avail)),
                "exc_episodes": len(b["excursions"]),
                "exc_episodes_alarm": int(sum(pair[s:e + 1].any() for s, e in b["excursions"])),
                "exc_episodes_retained_mode": int(sum(exc_mode[s:e + 1].any() for s, e in b["excursions"]))})
    for ty, tm in b["typemask"].items():
        out[f"fault_avail_s_{ty}"] = int(np.sum(tm & avail))
        out[f"fault_abs_err_{ty}"] = float(np.sum(err[tm & avail]))
    return out


def clopper_pearson(k, n, a=0.05):
    if n == 0:
        return np.nan, np.nan
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def summarise(evals, rng_boot):
    """Point estimates and batch-bootstrap intervals for one detector on the test batches."""
    nb = len(evals)
    types = TYPES

    def stats(ix):
        ev = [evals[i] for i in ix]
        out = {}
        for ty in types:
            ds = [d for e in ev for d in e["det"] if ty == "all" or d["type"] == ty]
            out[f"sens_{ty}"] = np.mean([d["detected"] for d in ds]) if ds else np.nan
            out[f"excmode_{ty}"] = np.mean([d["exc_mode"] and not d["detected"] for d in ds]) if ds else np.nan
            dl = [d["delay"] for d in ds if d["detected"]]
            out[f"delay_med_{ty}"] = np.median(dl) if dl else np.nan
            out[f"delay_p90_{ty}"] = np.percentile(dl, 90) if dl else np.nan
            iso = [d["iso_ok"] for d in ds if d["detected"] and not np.isnan(d["iso_ok"])]
            out[f"iso_{ty}"] = np.mean(iso) if iso else np.nan
            if ty != "all":
                fa_t = sum(e[f"fault_avail_s_{ty}"] for e in ev)
                out[f"mae_fault_{ty}"] = sum(e[f"fault_abs_err_{ty}"] for e in ev) / fa_t if fa_t else np.nan
        nh =sum(e["nominal_s"] for e in ev) / 3600.0
        out["fa_per_h"] = sum(e["fa_events"] for e in ev) / nh
        out["fa_exc_per_h"] = sum(e["fa_events_exc"] for e in ev) / nh
        out["fa_prob"] = sum(e["nominal_alarm_s"] for e in ev) / sum(e["nominal_s"] for e in ev)
        fa = sum(e["fault_avail_s"] for e in ev)
        out["mae_fault"] = sum(e["fault_abs_err"] for e in ev) / fa if fa else np.nan
        out["fault_avail"] = fa / sum(e["fault_s"] for e in ev)
        out["mae_nominal"] = sum(e["nom_abs_err"] for e in ev) / sum(e["nom_avail_s"] for e in ev)
        xs = sum(e["exc_s"] for e in ev)
        out["exc_withheld"] = sum(e["exc_withheld_s"] for e in ev) / xs if xs else np.nan
        xe = sum(e["exc_episodes"] for e in ev)
        out["exc_alarm"] = sum(e["exc_episodes_alarm"] for e in ev) / xe if xe else np.nan
        out["exc_mode"] = sum(e["exc_episodes_retained_mode"] for e in ev) / xe if xe else np.nan
        return out

    point = stats(range(nb))
    boots = [stats(rng_boot.integers(0, nb, nb)) for _ in range(N_BOOT)]
    ci = {k: (np.nanpercentile([bt[k] for bt in boots], 2.5), np.nanpercentile([bt[k] for bt in boots], 97.5))
          for k in point}
    counts = {}
    for ty in types:
        ds = [d for e in evals for d in e["det"] if ty == "all" or d["type"] == ty]
        counts[ty] = (sum(d["detected"] for d in ds), len(ds), sum(d["detected"] for d in ds))
    return point, ci, counts


# ----------------------------------------------------------------------------------------------- main
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN bootstrap slices (no detections)
    t0 = time.time()
    dev = [generate_batch(SEED_DEV + i) for i in range(N_DEV)]
    test = [generate_batch(SEED_TEST + i) for i in range(N_TEST)]
    fit_on_development(dev)
    rows = []

    def row(**kw):
        base = {"record_type": "", "detector": "", "fault_type": "", "metric": "", "estimate": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan, "interval_method": "", "n_units": np.nan, "unit": "",
                "threshold_m": np.nan, "threshold_unit": "", "split": "", "evidence_label": EVID}
        base.update(kw)
        rows.append(base)

    # generator description
    for split, bs in (("development", dev), ("test", test)):
        nf = {ty: sum(1 for b in bs for f in b["faults"] if f["type"] == ty) for ty in TYPES[1:]}
        row(record_type="design", split=split, metric="batches", estimate=len(bs), unit="batches of 2 h")
        for ty, v in nf.items():
            row(record_type="design", split=split, fault_type=ty, metric="fault_episodes", estimate=v, unit="episodes")
        row(record_type="design", split=split, metric="genuine_process_excursions",
            estimate=sum(len(b["excursions"]) for b in bs), unit="episodes")
        row(record_type="design", split=split, metric="nominal_hours",
            estimate=round(sum(b["nominal"].sum() for b in bs) / 3600.0, 2), unit="h")
    row(record_type="fitted_on_development", detector="gate", metric="sigma_r_robust", estimate=Fitted.sigma_r,
        unit="K", split="development")
    row(record_type="fitted_on_development", detector="gate", metric="model_lag_tau_hat", estimate=Fitted.tau_hat,
        unit="s", split="development")

    sc_dev = {d: [scores(d, b) for b in dev] for d in DETECTORS}
    sc_test = {d: [scores(d, b) for b in test] for d in DETECTORS}
    selected = {}
    for d in DETECTORS:
        grid = GRIDS[d]
        sel = None
        for m in grid:
            ev = [evaluate(b, *s, m, full=False) for b, s in zip(dev, sc_dev[d])]
            fa = sum(e["fa_events"] for e in ev) / (sum(e["nominal_s"] for e in ev) / 3600.0)
            det = [x["detected"] for e in ev for x in e["det"]]
            if sel is None and fa <= B_FA:
                sel = (m, fa, np.mean(det))
            evt = [evaluate(b, *s, m, full=False) for b, s in zip(test, sc_test[d])]
            fat = sum(e["fa_events"] for e in evt) / (sum(e["nominal_s"] for e in evt) / 3600.0)
            dett = [x["detected"] for e in evt for x in e["det"]]
            row(record_type="sweep", detector=d, fault_type="all", metric="sensitivity_test", estimate=np.mean(dett),
                threshold_m=m, threshold_unit=UNITS[d], split="test")
            row(record_type="sweep", detector=d, fault_type="all", metric="false_alarm_events_per_nominal_h_test",
                estimate=fat, threshold_m=m, threshold_unit=UNITS[d], split="test")
            row(record_type="sweep", detector=d, fault_type="all", metric="false_alarm_events_per_nominal_h_dev",
                estimate=fa, threshold_m=m, threshold_unit=UNITS[d], split="development")
        selected[d] = sel[0]
        row(record_type="selected_threshold", detector=d, metric="threshold_m", estimate=sel[0], unit=UNITS[d],
            threshold_m=sel[0], threshold_unit=UNITS[d], split="development")
        row(record_type="selected_threshold", detector=d, metric="false_alarm_events_per_nominal_h_dev",
            estimate=sel[1], threshold_m=sel[0], threshold_unit=UNITS[d], split="development")
        row(record_type="selected_threshold", detector=d, fault_type="all", metric="sensitivity_dev",
            estimate=sel[2], threshold_m=sel[0], threshold_unit=UNITS[d], split="development")

    rng_boot = np.random.default_rng(SEED_BOOT)
    names = {"fa_per_h": ("false_alarm_events_per_nominal_h", "events/h"),
             "fa_exc_per_h": ("false_alarm_events_during_genuine_excursions_per_nominal_h", "events/h"),
             "fa_prob": ("false_alarm_probability_per_nominal_sample", "fraction"),
             "mae_fault": ("post_gate_mean_abs_error_during_fault_episodes", "K"),
             "fault_avail": ("output_available_during_fault_episodes", "fraction"),
             "mae_nominal": ("post_gate_mean_abs_error_nominal", "K"),
             "exc_withheld": ("genuine_excursion_samples_withheld", "fraction"),
             "exc_alarm": ("genuine_excursion_episodes_with_alarm", "fraction"),
             "exc_mode": ("genuine_excursion_episodes_flagged_as_process_excursion", "fraction")}
    for d in DETECTORS:
        m = selected[d]
        evals = [evaluate(b, *s, m, full=True) for b, s in zip(test, sc_test[d])]
        point, ci, counts = summarise(evals, rng_boot)
        for ty in TYPES:
            k, n, _ = counts[ty]
            lo, hi = ci[f"sens_{ty}"]
            row(record_type="test_metric", detector=d, fault_type=ty, metric="sensitivity",
                estimate=point[f"sens_{ty}"], ci95_low=lo, ci95_high=hi, interval_method="batch_bootstrap_percentile",
                n_units=n, unit="fraction of episodes", threshold_m=m, threshold_unit=UNITS[d], split="test")
            cl, chh = clopper_pearson(k, n)
            row(record_type="test_metric", detector=d, fault_type=ty, metric="sensitivity",
                estimate=point[f"sens_{ty}"], ci95_low=cl, ci95_high=chh, interval_method="clopper_pearson_episodes",
                n_units=n, unit="fraction of episodes", threshold_m=m, threshold_unit=UNITS[d], split="test")
            for key, nm in ((f"delay_med_{ty}", "detection_delay_median"), (f"delay_p90_{ty}", "detection_delay_p90")):
                row(record_type="test_metric", detector=d, fault_type=ty, metric=nm, estimate=point[key],
                    ci95_low=ci[key][0], ci95_high=ci[key][1], interval_method="batch_bootstrap_percentile",
                    n_units=k, unit="s", threshold_m=m, threshold_unit=UNITS[d], split="test")
            if ty != "all":
                key = f"mae_fault_{ty}"
                row(record_type="test_metric", detector=d, fault_type=ty, metric="post_gate_mean_abs_error_in_episodes",
                    estimate=point[key], ci95_low=ci[key][0], ci95_high=ci[key][1],
                    interval_method="batch_bootstrap_percentile", n_units=n, unit="K",
                    threshold_m=m, threshold_unit=UNITS[d], split="test")
            if d == "gate":
                row(record_type="test_metric", detector=d, fault_type=ty,
                    metric="missed_episodes_retained_as_process_excursion", estimate=point[f"excmode_{ty}"],
                    ci95_low=ci[f"excmode_{ty}"][0], ci95_high=ci[f"excmode_{ty}"][1],
                    interval_method="batch_bootstrap_percentile", n_units=n, unit="fraction of episodes",
                    threshold_m=m, threshold_unit=UNITS[d], split="test")
            if d in ("fixed", "hampel", "gate"):
                row(record_type="test_metric", detector=d, fault_type=ty, metric="correct_isolation_at_first_alarm",
                    estimate=point[f"iso_{ty}"], ci95_low=ci[f"iso_{ty}"][0], ci95_high=ci[f"iso_{ty}"][1],
                    interval_method="batch_bootstrap_percentile", n_units=k, unit="fraction of detected episodes",
                    threshold_m=m, threshold_unit=UNITS[d], split="test")
        for key, (nm, unit) in names.items():
            if key == "exc_mode" and d != "gate":
                continue
            row(record_type="test_metric", detector=d, fault_type="na", metric=nm, estimate=point[key],
                ci95_low=ci[key][0], ci95_high=ci[key][1], interval_method="batch_bootstrap_percentile",
                n_units=N_TEST, unit=unit, threshold_m=m, threshold_unit=UNITS[d], split="test")
    df = pd.DataFrame(rows)
    df.insert(2, "detector_label", df["detector"].map(LABELS).fillna(""))
    for c in ("estimate", "ci95_low", "ci95_high"):
        df[c] = df[c].astype(float).round(6)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows) in {time.time() - t0:.0f} s")
    show = df[(df.record_type == "test_metric") & (df.interval_method != "clopper_pearson_episodes")]
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(df[df.record_type.isin(["design", "fitted_on_development", "selected_threshold"])]
              [["record_type", "detector", "fault_type", "metric", "estimate", "split"]].to_string())
        print(show[["detector", "fault_type", "metric", "estimate", "ci95_low", "ci95_high", "n_units"]].to_string())


if __name__ == "__main__":
    main()
