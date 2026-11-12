"""Executed joint state-and-parameter EKF and UKF on a reduced UHT heating-section fouling model (Section 5.10.4).

This is an executed example on a reduced model with generated trajectories. It does not re-estimate, replace or
rank the prescribed observer quantities of results/Ch05_Observer_Comparison_and_WCET_Benchmark.csv (Table 5.5),
it runs no MHE or particle filter, and it is not validation against plant data.

Generator ("plant", one 8-h run per seed, sampling interval 10 s, 2880 steps; declared values from
data/kinetic_and_transport_parameters.csv: milk flow 4.29 kg/s (PD-008), inlet 80 degC (PD-009), outlet set-point
138.5 degC (PD-010), terminal fouling resistance 3.2e-4 m2 K/W at 8 h for the nominal run (PD-013), outlet-
temperature noise SD 0.22 K (PD-015); clean overall coefficient U0 = 2850 W/(m2 K) taken from the UHT scenario of
Chapter 10 (PD-072); every other value below is assigned here):
  heating section lumped with an effectiveness relation, T_ss = T_h - (T_h - T_in) exp(-U A / (m c_p)),
      1/U = 1/U0 + R_f, c_p = 3900 J/(kg K), A sized so that the clean section needs T_h = 141.0 degC;
  outlet dynamics dT_out/dt = (T_ss - T_out)/20 s plus process noise SD 0.03 K per step;
  Kern-Seaton-type fouling (adapted from Eqs. (5.23a) and (5.45c)): dR_f/dt = k_dep(t) - k_rem R_f, k_rem = 0.1 1/h,
      k_dep(t) = k_run exp(-(E/R)(1/T_w - 1/T_w,ref)), E = 60 kJ/mol, T_w = (T_h + T_out)/2, k_run lognormal around
      the nominal value (SD of log 0.25); the nominal k_run is calibrated so that the nominal run reaches PD-013;
  inlet temperature AR(1) around 80 degC (SD 0.3 K, 30 min), true flow per run N(4.29, 0.04) kg/s (the observers
      assume 4.29), initial deposit |N(0, 0.05e-4)| m2 K/W;
  PI control of the heating-medium temperature T_h on the measured outlet (gain 0.5, integral time 60 s);
  measurements: T_out (SD 0.22 K), T_h and T_in (SD 0.10 K each).
Observers (identical information, model, noise settings and initialisation; tuning assigned a priori, not optimised on
these runs): augmented state x = [T_out, R_f, k_dep] with a random-walk deposition parameter (the Arrhenius
dependence and the true flow are unknown to them: deliberate structural mismatch);
  EKF  continuous-discrete joint EKF with analytic Jacobians, Joseph-form update;
  UKF  additive-noise unscented Kalman filter (alpha 1, beta 2, kappa 0; 7 sigma points);
  open loop  nominal model with k_dep fixed at the nominal value, no measurement update (reference only).
  Q = diag(0.10^2 K^2, (0.01e-4)^2, (0.3 k_nom)^2/2880) per step, R = 0.22^2 K^2,
  x0 = [first measurement, 0, k_nom], P0 = diag(0.22^2, (0.1e-4)^2, (0.35 k_nom)^2).
Metrics over 100 runs (seeds 5000-5099): RMSE of R_f over each run and at 8 h, relative error of the deposition
parameter over the last 4 h, RMSE of the filtered outlet temperature, NEES of the full augmented state (average
over runs per step, compared with the two-sided 95% chi-square band for 3 x 100 degrees of freedom), empirical
coverage of the nominal 95% interval of R_f, NIS, and wall-clock time per filter step (prediction + update),
measured with time.perf_counter_ns in this Python/NumPy implementation on the machine that runs the script:
wall-clock on a desktop PC, not WCET and not a statement about embedded hardware.
Intervals: percentile bootstrap over runs (2000 resamples, seed 12345).
Control run (record_type control_matched_generator): the same seeds with the Arrhenius term removed and the flow
set to the value assumed by the observers, to show how much of the inconsistency comes from that mismatch.
Local observability (record_type observability_run0): rank and condition number of [H; HF; HFF] of the linearised
augmented model at hourly points of run 5000 (scaled coordinates); a local check, not a proof of observability.

Evidence label: executed example on a reduced model with generated data; not a comparison of all listed
observers; not validation.

Output: Ch05_Executed_Observer_Dossier.csv (next to this script): design, per-run, summary and trajectory records.
Run:    python benchmark_ch05_observer_example.py      (about 1-3 min; numpy, scipy, pandas only)
"""
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

OUT = Path(__file__).resolve().parent / "Ch05_Executed_Observer_Dossier.csv"
EVID = "executed_example_reduced_model_generated_data_not_validation"

# declared (PD-xxx) and assigned values
M_DOT, T_IN, T_SP, RF_8H, SIG_T = 4.29, 80.0, 138.5, 3.2e-4, 0.22
U0, CP, TH_CLEAN = 2850.0, 3900.0, 141.0
TAU_HX, K_REM, E_DEP, RGAS = 20.0, 0.1, 60e3, 8.314462618
DT, N_STEPS, N_RUNS, SEED0 = 10.0, 2880, 100, 5000
SIG_TH, SIG_TIN, SIG_PROC = 0.10, 0.10, 0.03
SCALE = 1e-4                                   # R_f in units of 1e-4 m2 K/W, k_dep in 1e-4 m2 K/W per h
A_HX = -math.log((TH_CLEAN - T_SP) / (TH_CLEAN - T_IN)) * M_DOT * CP / U0
A_T = math.exp(-DT / TAU_HX)
A_R = math.exp(-K_REM * DT / 3600.0)
TW_REF = (TH_CLEAN + T_SP) / 2 + 273.15
N_BOOT, SEED_BOOT = 2000, 12345


def t_ss(th, tin, rf, mdot=M_DOT):
    ntu = A_HX / (mdot * CP) / (1.0 / U0 + rf * SCALE)
    return th - (th - tin) * math.exp(-ntu)


def dtss_drf(th, tin, rf, mdot=M_DOT):
    c = A_HX / (mdot * CP)
    den = 1.0 / U0 + rf * SCALE
    ntu = c / den
    return (th - tin) * math.exp(-ntu) * (-c * SCALE / den ** 2)


def th_required(tin, rf, mdot):
    """Heating-medium temperature that holds the set-point at steady state."""
    ntu = A_HX / (mdot * CP) / (1.0 / U0 + rf * SCALE)
    e = math.exp(-ntu)
    return (T_SP - tin * e) / (1 - e)


def k_arrhenius(k_run, th, tout):
    tw = (th + tout) / 2 + 273.15
    return k_run * math.exp(-E_DEP / RGAS * (1.0 / tw - 1.0 / TW_REF))


def calibrate_k_nom():
    k = 0.5
    for _ in range(30):
        rf = 0.0
        for _s in range(N_STEPS):
            th = th_required(T_IN, rf, M_DOT)
            kd = k_arrhenius(k, th, T_SP)
            rinf = kd / K_REM
            rf = rinf + (rf - rinf) * A_R
        k *= (RF_8H / SCALE) / rf
    return k


K_NOM = calibrate_k_nom()


def generate_run(seed, matched=False):
    """matched=True: control generator without the Arrhenius term and with the flow assumed by the observers."""
    rng = np.random.default_rng(seed)
    k_run = K_NOM * math.exp(rng.normal(0.0, 0.25))
    mdot = rng.normal(M_DOT, 0.04)
    if matched:
        mdot = M_DOT
    phi = math.exp(-DT / 1800.0)
    tin = T_IN + rng.normal(0, 0.3)
    rf = abs(rng.normal(0.0, 0.05e-4)) / SCALE
    th = th_required(tin, rf, mdot)
    tout = t_ss(th, tin, rf, mdot)
    integ = th
    rec = {k: np.zeros(N_STEPS) for k in ("T", "R", "k", "Th", "Tin", "y", "Th_m", "Tin_m")}
    for s in range(N_STEPS):
        kd = k_run if matched else k_arrhenius(k_run, th, tout)
        y = tout + rng.normal(0, SIG_T)
        rec["T"][s], rec["R"][s], rec["k"][s] = tout, rf, kd
        rec["Th"][s], rec["Tin"][s], rec["y"][s] = th, tin, y
        rec["Th_m"][s] = th + rng.normal(0, SIG_TH)
        rec["Tin_m"][s] = tin + rng.normal(0, SIG_TIN)
        # propagate plant over one interval with the inputs held
        tss = t_ss(th, tin, rf, mdot)
        tout = tss + (tout - tss) * A_T + rng.normal(0, SIG_PROC)
        rinf = kd / K_REM
        rf = rinf + (rf - rinf) * A_R
        tin = T_IN + (tin - T_IN) * phi + rng.normal(0, 0.3 * math.sqrt(1 - phi ** 2))
        # PI controller on the measured outlet temperature (acts for the next interval)
        e = T_SP - y
        integ += 0.5 * e * DT / 60.0
        th = min(160.0, max(130.0, integ + 0.5 * e))
    return rec


# ------------------------------------------------------------------------------------------- observer model
def fx(x, th, tin):
    T, R, k = x
    tss = t_ss(th, tin, R)
    rinf = k / K_REM
    return np.array([tss + (T - tss) * A_T, rinf + (R - rinf) * A_R, k])


def jac(x, th, tin):
    T, R, k = x
    return np.array([[A_T, (1 - A_T) * dtss_drf(th, tin, R), 0.0],
                     [0.0, A_R, (1 - A_R) / K_REM],
                     [0.0, 0.0, 1.0]])


H = np.array([[1.0, 0.0, 0.0]])
Q = np.diag([0.10 ** 2, 0.01 ** 2, (0.3 * K_NOM) ** 2 / N_STEPS])
R_MEAS = np.array([[SIG_T ** 2]])


def x0_P0(y0):
    return np.array([y0, 0.0, K_NOM]), np.diag([SIG_T ** 2, 0.1 ** 2, (0.35 * K_NOM) ** 2])


def kf_update(xp, Pp, y):
    S = H @ Pp @ H.T + R_MEAS
    K = Pp @ H.T / S[0, 0]
    nu = y - xp[0]
    x = xp + K[:, 0] * nu
    IKH = np.eye(3) - K @ H
    P = IKH @ Pp @ IKH.T + K @ R_MEAS @ K.T
    return x, P, nu * nu / S[0, 0]


def run_ekf(rec):
    x, P = x0_P0(rec["y"][0])
    out = np.zeros((N_STEPS, 3)); Ps = np.zeros((N_STEPS, 3, 3)); nis = np.zeros(N_STEPS); tns = np.zeros(N_STEPS)
    for s in range(N_STEPS):
        t0 = time.perf_counter_ns()
        if s > 0:
            F = jac(x, rec["Th_m"][s - 1], rec["Tin_m"][s - 1])
            x = fx(x, rec["Th_m"][s - 1], rec["Tin_m"][s - 1])
            P = F @ P @ F.T + Q
        x, P, nis[s] = kf_update(x, P, rec["y"][s])
        tns[s] = time.perf_counter_ns() - t0
        out[s], Ps[s] = x, P
    return out, Ps, nis, tns


ALPHA, BETA, KAPPA, NX = 1.0, 2.0, 0.0, 3
LAM = ALPHA ** 2 * (NX + KAPPA) - NX
WM = np.full(2 * NX + 1, 1.0 / (2 * (NX + LAM))); WC = WM.copy()
WM[0] = LAM / (NX + LAM); WC[0] = WM[0] + (1 - ALPHA ** 2 + BETA)


def sigma_points(x, P):
    L = np.linalg.cholesky((NX + LAM) * P)
    return np.vstack([x, x + L.T, x - L.T])


def run_ukf(rec):
    x, P = x0_P0(rec["y"][0])
    out = np.zeros((N_STEPS, 3)); Ps = np.zeros((N_STEPS, 3, 3)); nis = np.zeros(N_STEPS); tns = np.zeros(N_STEPS)
    for s in range(N_STEPS):
        t0 = time.perf_counter_ns()
        if s > 0:
            X = sigma_points(x, P)
            Xp = np.array([fx(xi, rec["Th_m"][s - 1], rec["Tin_m"][s - 1]) for xi in X])
            x = WM @ Xp
            D = Xp - x
            P = (WC[:, None] * D).T @ D + Q
        X = sigma_points(x, P)
        Yp = X[:, 0]
        yhat = WM @ Yp
        dy = Yp - yhat
        Syy = WC @ (dy * dy) + R_MEAS[0, 0]
        Pxy = (WC[:, None] * (X - x)).T @ dy
        K = Pxy / Syy
        nu = rec["y"][s] - yhat
        x = x + K * nu
        P = P - np.outer(K, K) * Syy
        P = 0.5 * (P + P.T)
        nis[s] = nu * nu / Syy
        tns[s] = time.perf_counter_ns() - t0
        out[s], Ps[s] = x, P
    return out, Ps, nis, tns


def run_open_loop(rec):
    x = np.array([rec["y"][0], 0.0, K_NOM])
    out = np.zeros((N_STEPS, 3))
    for s in range(N_STEPS):
        if s > 0:
            x = fx(x, rec["Th_m"][s - 1], rec["Tin_m"][s - 1])
        out[s] = x
    return out


# ------------------------------------------------------------------------------------------- evaluation
LATE = slice(N_STEPS // 2, None)
BURN = N_STEPS // 16                                    # first 30 min excluded from consistency statistics


def evaluate_runs(matched):
    per = {o: {m: [] for m in ("rmse_R", "err_R_8h", "relerr_k_late", "rmse_T", "cover95_R", "nis_mean")}
           for o in ("EKF", "UKF", "open_loop")}
    nees_sum = {o: np.zeros(N_STEPS) for o in ("EKF", "UKF")}
    times = {o: [] for o in ("EKF", "UKF")}
    traj = None
    for i in range(N_RUNS):
        rec = generate_run(SEED0 + i, matched=matched)
        truth = np.vstack([rec["T"], rec["R"], rec["k"]]).T
        res = {"EKF": run_ekf(rec), "UKF": run_ukf(rec)}
        ol = run_open_loop(rec)
        for o in ("EKF", "UKF", "open_loop"):
            est = ol if o == "open_loop" else res[o][0]
            e = est - truth
            per[o]["rmse_R"].append(math.sqrt(np.mean(e[:, 1] ** 2)))
            per[o]["err_R_8h"].append(abs(e[-1, 1]))
            per[o]["relerr_k_late"].append(math.sqrt(np.mean((e[LATE, 2] / truth[LATE, 2]) ** 2)))
            per[o]["rmse_T"].append(math.sqrt(np.mean(e[:, 0] ** 2)))
            if o != "open_loop":
                _, Ps, nis, tns = res[o]
                sdR = np.sqrt(Ps[:, 1, 1])
                per[o]["cover95_R"].append(np.mean(np.abs(e[BURN:, 1]) <= 1.96 * sdR[BURN:]))
                per[o]["nis_mean"].append(np.mean(nis[BURN:]))
                nees_sum[o] += np.einsum("ti,ti->t", e, np.linalg.solve(Ps, e[:, :, None])[:, :, 0])
                times[o].append(tns[1:])
            else:
                per[o]["cover95_R"].append(np.nan)
                per[o]["nis_mean"].append(np.nan)
        if i == 0:
            traj = (rec, res, ol)
    return per, nees_sum, times, traj


def local_observability(rec, est):
    """Rank and condition number of [H; HF; HF F] of the linearised augmented model at hourly points of one run
    (scaled coordinates of this script: K, 1e-4 m2 K/W, 1e-4 m2 K/W per h)."""
    out = []
    for s in range(360, N_STEPS, 360):
        F = jac(est[s], rec["Th_m"][s], rec["Tin_m"][s])
        O = np.vstack([H, H @ F, H @ F @ F])
        sv = np.linalg.svd(O, compute_uv=False)
        out.append((s * DT / 3600.0, int(np.sum(sv > sv[0] * 1e-12)), sv[0] / sv[-1]))
    return out


def main():
    t_start = time.time()
    rows = []

    def row(**kw):
        base = {"record_type": "", "observer": "", "metric": "", "estimate": np.nan, "ci95_low": np.nan,
                "ci95_high": np.nan, "interval_method": "", "n_units": np.nan, "unit": "", "run": np.nan,
                "time_h": np.nan, "evidence_label": EVID}
        base.update(kw)
        rows.append(base)

    row(record_type="design", metric="k_dep_nominal_calibrated_to_PD013", estimate=K_NOM * SCALE, unit="m2 K/W per h")
    row(record_type="design", metric="heating_area_assigned", estimate=A_HX, unit="m2")
    row(record_type="design", metric="runs", estimate=N_RUNS, unit="runs of 8 h")
    row(record_type="design", metric="steps_per_run", estimate=N_STEPS, unit="steps of 10 s")
    row(record_type="design", metric="platform", unit=f"{platform.platform()} | {platform.processor()} | "
                                                       f"Python {platform.python_version()} | NumPy {np.__version__}")
    units = {"rmse_R": "1e-4 m2 K/W", "err_R_8h": "1e-4 m2 K/W", "relerr_k_late": "fraction (RMS relative error)",
             "rmse_T": "K", "cover95_R": "fraction of run-steps after 30 min", "nis_mean": "mean NIS (1 dof)"}
    lo, hi = chi2.ppf(0.025, 3 * N_RUNS) / N_RUNS, chi2.ppf(0.975, 3 * N_RUNS) / N_RUNS
    rng = np.random.default_rng(SEED_BOOT)
    for variant, rtype in ((False, "summary"), (True, "control_matched_generator")):
        per, nees_sum, times, traj = evaluate_runs(variant)
        if not variant:
            for o in per:
                for i in range(N_RUNS):
                    for m, unit in (("rmse_R", "1e-4 m2 K/W"), ("relerr_k_late", "fraction")):
                        row(record_type="per_run", observer=o, metric=m, estimate=per[o][m][i], unit=unit,
                            run=SEED0 + i)
        for o in per:
            for m, vals in per[o].items():
                v = np.asarray(vals, float)
                if np.all(np.isnan(v)):
                    continue
                boots = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(N_BOOT)]
                row(record_type=rtype, observer=o, metric=m + "_mean_over_runs", estimate=np.mean(v),
                    ci95_low=np.percentile(boots, 2.5), ci95_high=np.percentile(boots, 97.5),
                    interval_method="bootstrap_over_runs", n_units=len(v), unit=units[m])
                row(record_type=rtype, observer=o, metric=m + "_p95_over_runs", estimate=np.percentile(v, 95),
                    n_units=len(v), unit=units[m])
        for o in ("EKF", "UKF"):
            anees = nees_sum[o] / N_RUNS
            row(record_type=rtype, observer=o, metric="ANEES_mean_after_30min", estimate=np.mean(anees[BURN:]),
                n_units=N_RUNS, unit="expected 3 for a consistent filter")
            row(record_type=rtype, observer=o, metric="ANEES_fraction_of_steps_inside_95pct_band",
                estimate=np.mean((anees[BURN:] >= lo) & (anees[BURN:] <= hi)), ci95_low=lo, ci95_high=hi,
                interval_method="band_limits_chi2_3x100_dof_over_100", n_units=N_STEPS - BURN,
                unit="fraction of steps")
            row(record_type=rtype, observer=o, metric="ANEES_fraction_of_steps_above_band",
                estimate=np.mean(anees[BURN:] > hi), n_units=N_STEPS - BURN, unit="fraction of steps")
            if not variant:
                tt = np.concatenate(times[o]) / 1000.0
                for qq, nm in ((50, "median"), (99, "p99"), (99.9, "p99_9")):
                    row(record_type=rtype, observer=o, metric=f"wall_clock_per_step_{nm}",
                        estimate=np.percentile(tt, qq), n_units=tt.size,
                        unit="microseconds (desktop PC, Python/NumPy; not WCET)")
                row(record_type=rtype, observer=o, metric="wall_clock_per_step_max_observed", estimate=tt.max(),
                    n_units=tt.size, unit="microseconds (desktop PC, Python/NumPy; not WCET)")
        if variant:
            continue
        rec, res, ol = traj
        obs = local_observability(rec, res["EKF"][0])
        for th_, rk, cn in obs:
            row(record_type="observability_run0", observer="EKF", metric="rank_of_local_observability_matrix",
                estimate=rk, unit="of 3", run=SEED0, time_h=th_)
            row(record_type="observability_run0", observer="EKF", metric="condition_number_scaled_coordinates",
                estimate=cn, unit="-", run=SEED0, time_h=th_)
        for s in range(0, N_STEPS, 180):
            th = s * DT / 3600.0
            row(record_type="trajectory_run0", observer="truth", metric="R_f", estimate=rec["R"][s],
                unit="1e-4 m2 K/W", run=SEED0, time_h=th)
            row(record_type="trajectory_run0", observer="truth", metric="k_dep", estimate=rec["k"][s],
                unit="1e-4 m2 K/W per h", run=SEED0, time_h=th)
            row(record_type="trajectory_run0", observer="truth", metric="T_h", estimate=rec["Th"][s], unit="degC",
                run=SEED0, time_h=th)
            row(record_type="trajectory_run0", observer="open_loop", metric="R_f", estimate=ol[s, 1],
                unit="1e-4 m2 K/W", run=SEED0, time_h=th)
            for o in ("EKF", "UKF"):
                est, Ps = res[o][0], res[o][1]
                sd = math.sqrt(Ps[s, 1, 1])
                row(record_type="trajectory_run0", observer=o, metric="R_f", estimate=est[s, 1],
                    ci95_low=est[s, 1] - 1.96 * sd, ci95_high=est[s, 1] + 1.96 * sd,
                    interval_method="filter_covariance", unit="1e-4 m2 K/W", run=SEED0, time_h=th)
                row(record_type="trajectory_run0", observer=o, metric="k_dep", estimate=est[s, 2],
                    unit="1e-4 m2 K/W per h", run=SEED0, time_h=th)
    df = pd.DataFrame(rows)
    for c in ("estimate", "ci95_low", "ci95_high"):
        df[c] = df[c].astype(float).round(6)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows) in {time.time() - t_start:.0f} s; k_nom = {K_NOM * SCALE:.4e}")
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(df[~df.record_type.isin(["per_run", "trajectory_run0"])]
              [["record_type", "observer", "metric", "estimate", "ci95_low", "ci95_high", "time_h"]].to_string())


if __name__ == "__main__":
    main()
