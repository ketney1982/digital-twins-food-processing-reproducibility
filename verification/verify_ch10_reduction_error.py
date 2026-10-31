"""Reduction-error verification and resolved re-evaluation of the Chapter 10 retort policies (Scenario IA).

The scenario runner (run_all_benchmarks.simulate_single_retort_batch) advances three lumped temperature states.
This script

  1. re-implements that reduced model unchanged for all four archived policies (two constant-retort-temperature
     (CRT) schedules, the assigned variable-retort-temperature (VRT) schedule and the adaptive surrogate) and
     reproduces the archived 500-trial table (results/Ch10_Table_10_4_Scenario_IA_Retort_Benchmark.csv);
  2. evaluates every candidate policy with ONE resolved model: axisymmetric finite-volume conduction in the
     half-can, Crank-Nicolson, 36 x 50 cells, 3 s step, retort temperature imposed at the surface (Dirichlet),
     constant properties. All policies see the same 500 disturbance realisations (seed 42: diffusivity multiplier,
     initial temperature, z-value, 32-38 min steam dip) and the controllers the same measurement-noise realisation;
  3. states what each adaptive policy observes:
       privileged (archived surrogate): its own three-node state computed with the trial diffusivity and initial
           temperature, the trial z-value, the dip magnitude before it occurs (set-point raised by 0.14 |dip| from
           22 min) and, in the archived run, a dip attenuated to 22 %;
       feed-forward (realisable): measured retort temperature and measured initial temperature (assigned noise,
           SD 0.5 K, samples every 0.25 min), an internal resolved conduction model (18 x 25 cells) with the nominal
           diffusivity, F0 computed with z = 10 degC; no knowledge of the trial diffusivity, z-value or dip;
       probe (realisable): as feed-forward plus the noisy centre temperature of one instrumented sentinel can of
           the same batch (SD 0.5 K); the diffusivity is selected online from a bank of internal models
           (multipliers 0.70-1.30, step 0.02) by least squares on the probe history, the accumulated F0 is
           computed from the probe and the remaining lethality is predicted with the selected model.
     Both realisable controllers keep the heating shape of the policy (CRT 121.1 degC, or the VRT plateau and
     129.6 degC pulse) and decide every 0.25 min from 34 min when to end heating (CRT) or start the 103 degC
     phase (VRT): when the predicted terminal F0 reaches 8 min + m. Heating ends at the latest at 100 min (CRT)
     or the pulse at 78 min (VRT);
  4. computes the uncertainty margin each policy needs on the resolved model: the common hold time (CRT), pulse
     end (VRT) or lethality margin m (controllers) at which the empirical 1st percentile of F0 reaches 8 min,
     calibrated on all 500 trials and, as an out-of-sample check, by two-fold cross-fitting (odd/even trial ids).

Scope: model-to-model verification of the reduction and executed benchmark of the policies on the resolved
conduction model, for conduction heating only. Measurement noise, the probe and the controllers are assigned
scenario elements. Nothing here is validation against measured heat-penetration data, and no result may be used
to set an industrial sterilisation time.

Outputs (written next to this script):
  Ch10_Reduction_Error_Solver_Checks.csv   FV solver against the analytical series, grid refinement, modal form
  Ch10_Reduction_Error_Trials.csv          per-trial reduced and resolved indicators (all policies x 500 trials)
  Ch10_Reduction_Error_Summary.csv         per policy: bias, F0 percentiles, under-target counts with binomial
                                           upper bounds, classification changes, retention, cook value, time,
                                           margin and cross-fitted counts

Run: python verify_ch10_reduction_error.py   (--quick: 60-trial smoke test without solver checks)
The resolved runs use small matrix products; with other jobs on the machine set OPENBLAS_NUM_THREADS=1 (or
OMP_NUM_THREADS / MKL_NUM_THREADS = 1). The stored outputs took 27 min single-threaded on a shared 16-core machine.
The vectorised modal solver reproduces the sparse solver (full_model) to 1e-10 min in F0 and 2e-5 percentage
points in thiamine retention for the two CRT schedules.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.special as sps
from scipy.linalg import eigh
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results"

R_CAN, H_CAN = 0.0419, 0.0579          # can radius and half-height (m), Appendix G PD-054/055
ALPHA_NOM, T0_NOM = 2.18e-7, 71.1      # m2/s, degC (PD-059/060)
D_THIA, Z_THIA, Z_COOK = 158.0, 25.5, 33.0
F0_REQ = 8.0
NR, NZ, DT_FV = 36, 50, 0.05           # FV grid and time step (min), see solver checks
LN10 = math.log(10.0)
R_STAR = 0.85 * R_CAN                  # radius of the reduced model's shell node (cook value)

# Assigned elements of the realisable controllers (scenario values, not instrument specifications)
NR_CTRL, NZ_CTRL = 18, 25              # internal model grid of the controllers
SIGMA_TR = SIGMA_IT = SIGMA_PROBE = 0.5  # measurement noise SD (K), white, one sample per 0.25 min
PROBE_BIAS = 0.5                       # constant probe offset (K) for the sensitivity row only
NOISE_SEED = 2026
ALPHA_BANK = np.round(np.arange(0.70, 1.3001, 0.02), 2)
T_DEC0, DT_DEC = 34.0, 0.25            # first decision time and decision interval (as in the runner)
T_FORCE = {"crt": 100.0, "vrt": 78.0}  # latest end of heating (CRT) or of the pulse (VRT)
TAIL = {"crt": 20.0, "vrt": 31.5}      # time from the decision to the end of the cycle
VRT_SET = 129.6


def retort_temperature(t, T_hold, t_hold, T0, dip):
    """CRT retort schedule exactly as in the scenario runner."""
    if t <= t_hold:
        T = T0 + (T_hold - T0) * (t / 6.0) if t <= 6.0 else T_hold
        if 32.0 <= t <= 38.0 and dip < 0.0:
            T += dip
        return T
    return 25.0 + (T_hold - 25.0) * math.exp(-(t - t_hold) / 3.6)


def reduced_model(T_hold, t_hold, alpha, T0, z_bot, dip):
    """Three-node reduced model (CRT branch of simulate_single_retort_batch)."""
    lam1 = (2.4048255 / R_CAN) ** 2 + (math.pi / (2.0 * H_CAN)) ** 2
    rate_c = 0.842 * alpha * lam1 * 60.0
    dt, t_cool = 0.25, 20.0
    Ts = Tm = Tc = T0
    F0 = ln_core = ln_shell = C0 = 0.0
    step = 0
    while True:
        step += 1
        t = step * dt
        Tr = retort_temperature(t, T_hold, t_hold, T0, dip)
        rs, rm, rr = 3.35 * rate_c, 2.25 * rate_c, 1.68 * rate_c
        Ts = (Ts + rs * dt * Tr) / (1.0 + rs * dt)
        Tm = (Tm + rm * dt * (0.64 * Ts + 0.36 * max(Ts, Tm))) / (1.0 + rm * dt)
        Tc = (Tc + rr * dt * Tm) / (1.0 + rr * dt)
        if Tc > 90.0:
            F0 += 10.0 ** ((Tc - 121.1) / z_bot) * dt
        if Ts > 70.0:
            k_c = (math.log(10.0) / D_THIA) * 10.0 ** ((Tc - 121.1) / Z_THIA)
            k_s = (math.log(10.0) / D_THIA) * 10.0 ** ((Ts - 121.1) / Z_THIA)
            ln_core -= k_c * dt
            ln_shell -= k_s * dt
            C0 += 0.74 * 10.0 ** ((Ts - 100.0) / Z_COOK) * dt
        if t >= t_hold + t_cool:
            break
    cv = (0.28 * math.exp(ln_core) + 0.72 * math.exp(ln_shell)) * 100.0
    return F0, cv, C0


_UNIT_CACHE = {}


def unit_operator(nr, nz):
    """Finite-volume Laplacian (unit diffusivity, per second) on the (r, z) half-can; symmetry at z = 0
    and r = 0, Dirichlet retort temperature at r = R and z = H (half-cell distance to the boundary)."""
    key = (nr, nz)
    if key in _UNIT_CACHE:
        return _UNIT_CACHE[key]
    dr, dz = R_CAN / nr, H_CAN / nz
    rc = (np.arange(nr) + 0.5) * dr
    rf = np.arange(nr + 1) * dr
    n = nr * nz
    A = sp.lil_matrix((n, n))
    b = np.zeros(n)
    vols = np.zeros(n)
    for i in range(nr):
        for j in range(nz):
            k = i * nz + j
            V = rc[i] * dr * dz
            vols[k] = V
            ae, aw, az = rf[i + 1] * dz / dr, rf[i] * dz / dr, rc[i] * dr / dz
            if i < nr - 1:
                A[k, k + nz] += ae / V
                A[k, k] -= ae / V
            else:
                A[k, k] -= 2.0 * ae / V
                b[k] += 2.0 * ae / V
            if i > 0:
                A[k, k - nz] += aw / V
                A[k, k] -= aw / V
            if j < nz - 1:
                A[k, k + 1] += az / V
                A[k, k] -= az / V
            else:
                A[k, k] -= 2.0 * az / V
                b[k] += 2.0 * az / V
            if j > 0:
                A[k, k - 1] += az / V
                A[k, k] -= az / V
    out = (A.tocsc(), b, vols, rc)
    _UNIT_CACHE[key] = out
    return out


def full_model(T_hold, t_hold, alpha, T0, z_bot, dip, nr=NR, nz=NZ, dt=DT_FV):
    """Crank-Nicolson solution of axisymmetric conduction; indicators evaluated on the resolved field."""
    A1, b1, vols, rc = unit_operator(nr, nz)
    n = nr * nz
    Ad = A1 * (alpha * 60.0)                       # per minute
    bd = b1 * (alpha * 60.0)
    eye = sp.identity(n, format="csc")
    lu = spl.splu((eye - 0.5 * dt * Ad).tocsc())
    rhs = (eye + 0.5 * dt * Ad).tocsr()
    T = np.full(n, T0)
    F0 = C0 = 0.0
    ln_c = np.zeros(n)
    t, t_end = 0.0, t_hold + 20.0
    r_star = 0.85 * R_CAN                          # radius of the reduced model's shell node
    for _ in range(int(round(t_end / dt))):
        Tr0 = retort_temperature(t, T_hold, t_hold, T0, dip)
        Tr1 = retort_temperature(t + dt, T_hold, t_hold, T0, dip)
        T = lu.solve(rhs @ T + 0.5 * dt * bd * (Tr0 + Tr1))
        t += dt
        Tcore = T[0]                               # first cell centre (dr/2, dz/2), nearest to the geometric centre
        if Tcore > 90.0:
            F0 += 10.0 ** ((Tcore - 121.1) / z_bot) * dt
        k = np.where(T > 70.0, (math.log(10.0) / D_THIA) * 10.0 ** ((T - 121.1) / Z_THIA), 0.0)
        ln_c -= k * dt
        Ts = float(np.interp(r_star, rc, T.reshape(nr, nz)[:, 0]))
        if Ts > 70.0:
            C0 += 0.74 * 10.0 ** ((Ts - 100.0) / Z_COOK) * dt
    cv = float((np.exp(ln_c) * vols).sum() / vols.sum() * 100.0)
    return F0, cv, C0


def analytic_centre(t_min, T_hold, T0, alpha):
    """Centre temperature of a finite cylinder after a step in surface temperature (series solution)."""
    t = t_min * 60.0
    mu = sps.jn_zeros(0, 200)
    s_r = sum(2.0 / (mu[n] * sps.j1(mu[n])) * math.exp(-alpha * (mu[n] / R_CAN) ** 2 * t) for n in range(200))
    s_z = sum(2.0 * (-1) ** k / ((2 * k + 1) * math.pi / 2.0) *
              math.exp(-alpha * ((2 * k + 1) * math.pi / (2.0 * H_CAN)) ** 2 * t) for k in range(400))
    return T_hold - (T_hold - T0) * s_r * s_z


def solver_checks():
    rows = []
    T_hold, T0 = 121.1, T0_NOM
    for nr, nz, dt in [(18, 25, 0.1), (36, 50, 0.05), (72, 100, 0.025)]:
        A1, b1, _, _ = unit_operator(nr, nz)
        n = nr * nz
        eye = sp.identity(n, format="csc")
        Ad, bd = A1 * (ALPHA_NOM * 60.0), b1 * (ALPHA_NOM * 60.0)
        lu = spl.splu((eye - 0.5 * dt * Ad).tocsc())
        rhs = (eye + 0.5 * dt * Ad).tocsr()
        T, t = np.full(n, T0), 0.0
        marks = {}
        for _ in range(int(round(40.0 / dt))):
            T = lu.solve(rhs @ T + 0.5 * dt * bd * (2.0 * T_hold))
            t += dt
            for tt in (10, 20, 30, 40):
                if abs(t - tt) < dt / 2:
                    marks[tt] = T[0]
        F0, cv, c0 = full_model(121.1, 59.5, ALPHA_NOM, T0_NOM, 10.0, 0.0, nr, nz, dt)
        rows.append({"grid_nr": nr, "grid_nz": nz, "dt_min": dt,
                     "T_centre_10min": round(marks[10], 3), "T_centre_20min": round(marks[20], 3),
                     "T_centre_30min": round(marks[30], 3), "T_centre_40min": round(marks[40], 3),
                     "F0_nominal_CRT_min": round(F0, 3), "thiamine_retention_pct": round(cv, 3),
                     "surface_cook_C0_min": round(c0, 2)})
    rows.append({"grid_nr": "analytic", "grid_nz": "series", "dt_min": "",
                 **{f"T_centre_{tt}min": round(analytic_centre(tt, T_hold, T0, ALPHA_NOM), 3) for tt in (10, 20, 30, 40)},
                 "F0_nominal_CRT_min": "", "thiamine_retention_pct": "", "surface_cook_C0_min": ""})
    return rows


def load_trials():
    with open(DATA / "monte_carlo_disturbances_seed42.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ============================================================================================================
# Vectorised forms used for the policy comparison (identical arithmetic to the functions above)
# ============================================================================================================
def modal_basis(nr, nz):
    """Diagonalise the FV operator of unit_operator. On the rectangular (r, z) grid it is a Kronecker sum,
    A = A_r (x) I + I (x) A_z, with A_r symmetric in the weights r_i and A_z symmetric, so its eigenvectors are
    products of radial and axial eigenvectors. Crank-Nicolson is then exact mode by mode, trials with different
    diffusivities run as array rows, and the field is rebuilt as Phi_r Y Phi_z^T."""
    dr, dz = R_CAN / nr, H_CAN / nz
    rc = (np.arange(nr) + 0.5) * dr
    rf = np.arange(nr + 1) * dr
    Ar, br = np.zeros((nr, nr)), np.zeros(nr)
    for i in range(nr):
        if i < nr - 1:
            c = rf[i + 1] / (rc[i] * dr * dr)
            Ar[i, i + 1] += c
            Ar[i, i] -= c
        else:
            c = 2.0 * rf[nr] / (rc[i] * dr * dr)
            Ar[i, i] -= c
            br[i] += c
        if i > 0:
            c = rf[i] / (rc[i] * dr * dr)
            Ar[i, i - 1] += c
            Ar[i, i] -= c
    Az, bz = np.zeros((nz, nz)), np.zeros(nz)
    for j in range(nz):
        if j < nz - 1:
            Az[j, j + 1] += 1.0 / dz ** 2
            Az[j, j] -= 1.0 / dz ** 2
        else:
            Az[j, j] -= 2.0 / dz ** 2
            bz[j] += 2.0 / dz ** 2
        if j > 0:
            Az[j, j - 1] += 1.0 / dz ** 2
            Az[j, j] -= 1.0 / dz ** 2
    A1, b1, _, _ = unit_operator(nr, nz)              # consistency with the sparse operator
    dev = float(abs(sp.kron(Ar, sp.identity(nz)) + sp.kron(sp.identity(nr), Az) - A1).max() / abs(A1).max())
    dev = max(dev, float(np.abs(np.add.outer(br, bz).ravel() - b1).max() / np.abs(b1).max()))
    sr = np.sqrt(rc)
    lr, Qr = eigh((sr[:, None] * Ar) / sr[None, :] * 0.5 + ((sr[:, None] * Ar) / sr[None, :]).T * 0.5)
    lz, Qz = eigh(0.5 * (Az + Az.T))
    Pr, Pz = Qr / sr[:, None], Qz                     # field = Pr @ Y @ Pz.T
    proj = lambda F: ((Qr.T * sr[None, :]) @ F @ Qz).ravel()
    i0 = int(np.searchsorted(rc, R_STAR) - 1)
    w1 = (R_STAR - rc[i0]) / (rc[i0 + 1] - rc[i0])
    return {"lam": np.add.outer(lr, lz).ravel(), "Pr32": Pr.astype(np.float32), "PzT32": Pz.T.astype(np.float32),
            "beta": proj(np.add.outer(br, bz)), "y1": proj(np.ones((nr, nz))),
            "wc": np.outer(Pr[0], Pz[0]).ravel(), "ws": np.outer((1.0 - w1) * Pr[i0] + w1 * Pr[i0 + 1], Pz[0]).ravel(),
            "pr0": Pr[0].copy(), "prs": (1.0 - w1) * Pr[i0] + w1 * Pr[i0 + 1], "pz0": Pz[0].copy(),
            "vols": (rc[:, None] * dr * dz) * np.ones((1, nz)), "nr": nr, "nz": nz, "n": nr * nz, "dev": dev}


def cn_coeffs(B, alpha, dt=DT_FV):
    """CN amplification g and input gain h, shape (trials, modes): y+ = g y + h (u + u+)."""
    a60 = np.asarray(alpha, float) * 60.0
    mu = a60[:, None] * B["lam"][None, :]
    den = 1.0 - 0.5 * dt * mu
    return (1.0 + 0.5 * dt * mu) / den, 0.5 * dt * a60[:, None] * B["beta"][None, :] / den


def lethal(T, z):
    return np.where(T > 90.0, np.power(10.0, (T - 121.1) / z), 0.0)


def retort_profile(t, shape, T0, dip, T_set, t_sw, dipf=1.0, inclusive=True, t_pre_cool=None):
    """Retort temperature at times t (rows) for every trial (columns), and the heating indicator.
    crt: ramp from T0 to T_set over 6 min, hold until t_sw, cooling 25 + (T_set - 25) exp(-(t - t_sw)/3.6).
    vrt: ramp from T0 to 108 degC over 6 min, 108 degC until 22 min, T_set until t_sw, 103 degC until
         t_sw + 11.5 min (end of heating), cooling 25 + 78 exp(-(t - t_h)/3.5).
    The steam dip (dip * dipf) is added between 32 and 38 min while heating. t_pre_cool reproduces the archived
    adaptive branch when its pulse outlasts the nominal 60.5 min (cooling until the late switch)."""
    e = 1e-7                                       # times are multiples of 0.05 min; compare without round-off
    t = np.asarray(t, float)[:, None]
    T0, dip, T_set, t_sw = (np.broadcast_to(np.asarray(v, float), (np.size(T0),)) for v in (T0, dip, T_set, t_sw))
    if shape == "crt":
        t_h = t_sw
        Th = np.where(t <= 6.0 + e, T0 + (T_set - T0) * (t / 6.0), T_set)
        cool = 25.0 + (T_set - 25.0) * np.exp(np.minimum(-(t - t_h) / 3.6, 50.0))
    else:
        t_h = t_sw + 11.5
        pulse = (t <= t_sw + e) if inclusive else (t < t_sw - e)
        Th = np.where(t <= 6.0 + e, T0 + (108.0 - T0) * (t / 6.0),
                      np.where(t <= 22.0 + e, 108.0, np.where(pulse, T_set, 103.0)))
        cool = 25.0 + 78.0 * np.exp(np.minimum(-(t - t_h) / 3.5, 50.0))
    Th = Th + np.where((t >= 32.0 - e) & (t <= 38.0 + e) & (dip < 0.0), dip * dipf, 0.0)
    heat = t <= t_h + e
    T = np.where(heat, Th, cool)
    if t_pre_cool is not None:
        pre = (t > t_pre_cool + e) & (t < t_sw - e)
        T = np.where(pre, 25.0 + 78.0 * np.exp(np.minimum(-(t - t_pre_cool) / 3.5, 50.0)), T)
        heat = heat & ~pre
    return T, heat, t_h


def reduced_batch(alpha, T0, z, Tr, heat, t_end):
    """Three-node model of simulate_single_retort_batch (implicit Euler, 0.25 min) for given retort histories
    Tr[k] at t = 0.25 k; stops at the first step with t >= t_end. Returns F0, C_V, C0, exergy (kJ/can)."""
    dt = 0.25
    lam1 = (2.4048255 / R_CAN) ** 2 + (math.pi / (2.0 * H_CAN)) ** 2
    rate = 0.842 * alpha * lam1 * 60.0
    rs, rm, rr = 3.35 * rate, 2.25 * rate, 1.68 * rate
    n_end = np.ceil(t_end / dt - 1e-9).astype(int)
    Ts, Tm, Tc = T0.copy(), T0.copy(), T0.copy()
    F0, lnc, lns, C0, ex = (np.zeros(len(alpha)) for _ in range(5))
    kk = LN10 / D_THIA
    for k in range(1, int(n_end.max()) + 1):
        act = k <= n_end
        T_r = Tr[k]
        Ts_n = (Ts + rs * dt * T_r) / (1.0 + rs * dt)
        Tm_n = (Tm + rm * dt * (0.64 * Ts_n + 0.36 * np.maximum(Ts_n, Tm))) / (1.0 + rm * dt)
        Tc_n = (Tc + rr * dt * Tm_n) / (1.0 + rr * dt)
        Ts, Tm, Tc = np.where(act, Ts_n, Ts), np.where(act, Tm_n, Tm), np.where(act, Tc_n, Tc)
        F0 += np.where(act, lethal(Tc, z), 0.0) * dt
        sh = act & (Ts > 70.0)
        lnc -= np.where(sh, kk * np.power(10.0, (Tc - 121.1) / Z_THIA), 0.0) * dt
        lns -= np.where(sh, kk * np.power(10.0, (Ts - 121.1) / Z_THIA), 0.0) * dt
        C0 += np.where(sh, 0.74 * np.power(10.0, (Ts - 100.0) / Z_COOK), 0.0) * dt
        q = 0.0035 * np.maximum(0.0, T_r - 0.65 * Ts - 0.35 * Tc) + np.where(T_r < 110.0, 0.0825, 0.1225)
        ex += np.where(act & heat[k], (1.0 - 293.15 / (T_r + 273.15)) * q * dt * 60.0, 0.0)
    return F0, (0.28 * np.exp(lnc) + 0.72 * np.exp(lns)) * 100.0, C0, ex


def archived_adaptive(alpha, T0, z, dip, q_conf, dipf=0.22, T_set=VRT_SET, t_hold=60.5):
    """Vectorised copy of the compensated (adaptive) branch of simulate_single_retort_batch: three-node state with
    the trial parameters, rollout every 0.25 min from 34 min, target 8 + q_conf + max(0, (10 - z) 0.35),
    set-point raised by 0.14 |dip| and the dip scaled by dipf (0.22 in the archived run)."""
    dt, n = 0.25, len(alpha)
    lam1 = (2.4048255 / R_CAN) ** 2 + (math.pi / (2.0 * H_CAN)) ** 2
    rate = 0.842 * alpha * lam1 * 60.0
    rs, rm, rr = 3.35 * rate, 2.25 * rate, 1.68 * rate
    Ts, Tm, Tc = T0.copy(), T0.copy(), T0.copy()
    F0, lnc, lns, C0, ex = (np.zeros(n) for _ in range(5))
    T_eff = T_set + np.abs(dip) * 0.14
    t_heff, t_s = np.full(n, t_hold), np.full(n, np.nan)
    started, done = np.zeros(n, bool), np.zeros(n, bool)
    z_est = 0.75 * z + 0.25 * 10.0
    target = F0_REQ + q_conf + np.maximum(0.0, (10.0 - z) * 0.35)
    te = np.arange(1, int(round(31.5 / dt)) + 1) * dt
    Tp = np.where(te <= 11.5, 103.0, 25.0 + 78.0 * np.exp(-(te - 11.5) / 3.5))
    kk = LN10 / D_THIA
    step = 0
    while not done.all():
        step += 1
        t = step * dt
        live = ~done
        cand = np.where(live & ~started & (t >= 34.0))[0]
        if cand.size:
            a, b_, c, f = Ts[cand].copy(), Tm[cand].copy(), Tc[cand].copy(), F0[cand].copy()
            for Tq in Tp:
                a = (a + rs[cand] * dt * Tq) / (1.0 + rs[cand] * dt)
                b_ = (b_ + rm[cand] * dt * (0.64 * a + 0.36 * np.maximum(a, b_))) / (1.0 + rm[cand] * dt)
                c = (c + rr[cand] * dt * b_) / (1.0 + rr[cand] * dt)
                f += lethal(c, z_est[cand]) * dt
            ii = cand[(f >= target[cand]) | (t >= 78.0)]
            started[ii], t_s[ii], t_heff[ii] = True, t, t + 11.5
        hold = t <= t_heff
        Th = T0 + (108.0 - T0) * (t / 6.0) if t <= 6.0 else (np.full(n, 108.0) if t <= 22.0 else np.where(started, 103.0, T_eff))
        if 32.0 <= t <= 38.0:
            Th = Th + np.where(dip < 0.0, dip * dipf, 0.0)
        T_r = np.where(hold, Th, 25.0 + 78.0 * np.exp(-(t - t_heff) / 3.5))
        Ts_n = (Ts + rs * dt * T_r) / (1.0 + rs * dt)
        Tm_n = (Tm + rm * dt * (0.64 * Ts_n + 0.36 * np.maximum(Ts_n, Tm))) / (1.0 + rm * dt)
        Tc_n = (Tc + rr * dt * Tm_n) / (1.0 + rr * dt)
        Ts, Tm, Tc = np.where(live, Ts_n, Ts), np.where(live, Tm_n, Tm), np.where(live, Tc_n, Tc)
        F0 += np.where(live, lethal(Tc, z), 0.0) * dt
        sh = live & (Ts > 70.0)
        lnc -= np.where(sh, kk * np.power(10.0, (Tc - 121.1) / Z_THIA), 0.0) * dt
        lns -= np.where(sh, kk * np.power(10.0, (Ts - 121.1) / Z_THIA), 0.0) * dt
        C0 += np.where(sh, 0.74 * np.power(10.0, (Ts - 100.0) / Z_COOK), 0.0) * dt
        q = 0.0035 * np.maximum(0.0, T_r - 0.65 * Ts - 0.35 * Tc) + np.where(T_r < 110.0, 0.0825, 0.1225)
        ex += np.where(live & hold, (1.0 - 293.15 / (T_r + 273.15)) * q * dt * 60.0, 0.0)
        done = done | (live & (t >= t_heff + 20.0))
    return {"F0": F0, "CV": (0.28 * np.exp(lnc) + 0.72 * np.exp(lns)) * 100.0, "C0": C0, "ex": ex,
            "t_s": t_s, "t_h": t_heff, "T_eff": T_eff}


def plant_run(B, alpha, T0, z, U, heat, t_end):
    """Resolved model for all trials at once (CN in modal form, 0.05 min). U[k], heat[k] at t = 0.05 k.
    Returns F0 (centre cell), thiamine volume average, cook value at r = 0.85R (mid-height), exergy (kJ/can).
    Modal state in the layout (radial mode, trial, axial mode) so that the field is two plain matrix products."""
    n, nr, nz = len(alpha), B["nr"], B["nz"]
    g, h = (np.ascontiguousarray(c.reshape(n, nr, nz).transpose(1, 0, 2)) for c in cn_coeffs(B, alpha))
    y = T0[None, :, None] * B["y1"].reshape(nr, nz)[:, None, :]
    n_end = np.round(t_end / DT_FV).astype(int)
    F0, C0, ex = np.zeros(n), np.zeros(n), np.zeros(n)
    lnc = np.zeros((nr, n, nz), np.float32)
    kk, cz = np.float32(LN10 / D_THIA), np.float32(LN10 / Z_THIA)
    for k in range(int(n_end.max())):
        y *= g
        y += h * (U[k] + U[k + 1])[None, :, None]
        act = (k + 1) <= n_end
        Tc = (B["pr0"] @ y.reshape(nr, n * nz)).reshape(n, nz) @ B["pz0"]
        Ts = (B["prs"] @ y.reshape(nr, n * nz)).reshape(n, nz) @ B["pz0"]
        F0 += np.where(act, lethal(Tc, z), 0.0) * DT_FV
        C0 += np.where(act & (Ts > 70.0), 0.74 * np.power(10.0, (Ts - 100.0) / Z_COOK), 0.0) * DT_FV
        T_r = U[k + 1]
        q = 0.0035 * np.maximum(0.0, T_r - 0.65 * Ts - 0.35 * Tc) + np.where(T_r < 110.0, 0.0825, 0.1225)
        ex += np.where(act & heat[k + 1], (1.0 - 293.15 / (T_r + 273.15)) * q * DT_FV * 60.0, 0.0)
        T = ((B["Pr32"] @ y.astype(np.float32).reshape(nr, n * nz)).reshape(nr * n, nz) @ B["PzT32"]).reshape(nr, n, nz)
        rate = (T > 70.0) * (kk * np.exp((T - np.float32(121.1)) * cz))
        lnc -= rate * (np.float32(DT_FV) * act.astype(np.float32))[None, :, None]
    cv = np.einsum("ij,inj->n", B["vols"], np.exp(lnc.astype(float))) / B["vols"].sum() * 100.0
    return {"F0": F0, "CV": cv, "C0": C0, "ex": ex}


def core_run(B, alpha, T0, U, n_steps):
    """Centre temperature only (rows t = 0.05 k); T0 = None starts from zero (response to an input increment)."""
    g, h = cn_coeffs(B, alpha)
    y = np.zeros((len(alpha), B["n"])) if T0 is None else T0[:, None] * B["y1"][None, :]
    out = np.empty((n_steps + 1, len(alpha)))
    out[0] = 0.0 if T0 is None else T0
    for k in range(n_steps):
        y = g * y + h * (U[k] + U[k + 1])[:, None]
        out[k + 1] = y @ B["wc"]
    return out


def planned_tail(shape, T_set, dt=DT_FV):
    """Retort temperature planned after a decision at s = j dt, j = 0..J (crt: cooling; vrt: 103 degC, cooling)."""
    s = np.arange(int(round(TAIL[shape] / dt)) + 1) * dt
    if shape == "crt":
        return 25.0 + (T_set - 25.0) * np.exp(-s / 3.6)
    return np.where(s <= 11.5, 103.0, 25.0 + 78.0 * np.exp(-(s - 11.5) / 3.5))


def switch_table(Bp, tr, shape, T_set):
    """Resolved F0 of every trial for every switch time t_s on the 0.05 min grid between 34 min and T_FORCE.
    Exact by linearity: centre(t) = centre without switch + K(t - t_s), K = response to the input change."""
    J = int(round(TAIL[shape] / DT_FV))
    n_cap = int(round(T_FORCE[shape] / DT_FV))
    N = n_cap + J
    tg = np.arange(N + 1) * DT_FV
    U, _, _ = retort_profile(tg, shape, tr["T0"], tr["dip"], T_set, 1e6)
    core = core_run(Bp, tr["alpha"], tr["T0"], U, N)
    w = planned_tail(shape, T_set)
    D = np.where(np.arange(J + 1) == 0, 0.0, w - T_set)
    K = core_run(Bp, tr["alpha"], None, np.repeat(D[:, None], len(tr["alpha"]), 1), J)
    Lc = np.cumsum(lethal(core, tr["z"]), 0) * DT_FV          # Lc[n] includes steps 0..n; step 0 is below 90 degC
    n0 = int(round(T_DEC0 / DT_FV))
    tab = np.empty((n_cap - n0 + 1, len(tr["alpha"])))
    for i, ns in enumerate(range(n0, n_cap + 1)):
        tab[i] = Lc[ns] + (lethal(core[ns + 1:ns + J + 1] + K[1:], tr["z"]).sum(0)) * DT_FV
    return {"tab": tab, "n0": n0, "core": core, "U": U, "N": N}


def controller_signals(Bc, tr, shape, T_set, sw, noise):
    """Run the realisable controllers along the trajectory without switching and record, at each decision time,
    the predicted terminal F0 (accumulated + predicted remainder) of the feed-forward and probe controllers.
    Until a controller switches, its trajectory equals this one, so the switch time for any margin m follows.
    Controllers sample every 0.25 min and advance their internal model (CN, 18 x 25 cells) with that step."""
    n = len(tr["alpha"])
    dtc = DT_DEC
    stride = int(round(dtc / DT_FV))
    kc = int(round(T_FORCE[shape] / dtc))
    J = int(round(TAIL[shape] / dtc))
    pidx = np.arange(kc + 1) * stride
    u_meas = sw["U"][pidx] + SIGMA_TR * noise["tr"][:kc + 1, :n]
    it_meas = tr["T0"] + SIGMA_IT * noise["it"][:n]
    probe = sw["core"][pidx] + SIGMA_PROBE * noise["pr"][:kc + 1, :n]
    A = len(ALPHA_BANK)
    a_nom = int(np.argmin(np.abs(ALPHA_BANK - 1.0)))
    gs, hs = cn_coeffs(Bc, ALPHA_BANK * ALPHA_NOM, dt=dtc)         # (A, M)
    g3, h3 = gs[:, None, :], hs[:, None, :]
    y = np.repeat((it_meas[:, None] * Bc["y1"][None, :])[None], A, 0)   # (A, n, M)
    wc = Bc["wc"]
    w = planned_tail(shape, T_set, dtc)
    jj = np.arange(1, J + 1)
    G, f, e = [], [], []                                            # predicted centre: G y + f + e u0
    for a in range(A):
        G.append(wc[None, :] * gs[a][None, :] ** jj[:, None])
        x, xe, fa, ea = np.zeros(Bc["n"]), hs[a].copy(), np.empty(J), np.empty(J)
        for j in range(1, J + 1):
            x = gs[a] * x + hs[a] * ((w[j - 1] if j > 1 else 0.0) + w[j])
            fa[j - 1], ea[j - 1] = wc @ x, wc @ xe
            xe = gs[a] * xe
        f.append(fa)
        e.append(ea)

    def remainder(a, yv, u0):
        C = G[a] @ yv.T + f[a][:, None] + e[a][:, None] * u0[None, :]
        return lethal(C, 10.0).sum(0) * dtc

    k0 = int(round(T_DEC0 / dtc))
    nd = kc - k0 + 1
    P = {"ff": np.empty((nd, n)), "pr": np.empty((nd, n)), "pr_bias": np.empty((nd, n))}
    a_hat_rec = np.empty((nd, n), int)
    sse, sse_b = np.zeros((A, n)), np.zeros((A, n))
    F_ff, F_pr, F_prb = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(kc):
        y *= g3
        y += h3 * (u_meas[k] + u_meas[k + 1])[None, :, None]
        cb = y @ wc                                                  # (A, n)
        sse += (probe[k + 1][None, :] - cb) ** 2
        sse_b += (probe[k + 1][None, :] + PROBE_BIAS - cb) ** 2
        F_ff += lethal(cb[a_nom], 10.0) * dtc
        F_pr += lethal(probe[k + 1], 10.0) * dtc
        F_prb += lethal(probe[k + 1] + PROBE_BIAS, 10.0) * dtc
        if k + 1 < k0:
            continue
        d = k + 1 - k0
        u0 = u_meas[k + 1]
        P["ff"][d] = F_ff + remainder(a_nom, y[a_nom], u0)
        for key, s_, Facc in (("pr", sse, F_pr), ("pr_bias", sse_b, F_prb)):
            ah = np.argmin(s_, 0)
            rem = np.empty(n)
            for a in np.unique(ah):
                idx = np.where(ah == a)[0]
                rem[idx] = remainder(a, y[a][idx], u0[idx])
            P[key][d] = Facc + rem
            if key == "pr":
                a_hat_rec[d] = ah
    return P, a_hat_rec


def p1(x):
    return float(np.percentile(x, 1.0))


def trigger_times(P, m, shape):
    """First decision time with predicted terminal F0 >= 8 + m; otherwise the forced time."""
    hit = P >= F0_REQ + m
    d = np.where(hit.any(0), hit.argmax(0), P.shape[0] - 1)
    return T_DEC0 + DT_DEC * d


def f0_at(sw, ts):
    i = np.round(ts / DT_FV).astype(int) - sw["n0"]
    return sw["tab"][i, np.arange(len(ts))]


def calibrate_margin(P, sw, shape, idx):
    """Smallest margin m (0.005 min resolution) with empirical 1st percentile of resolved F0 >= 8 on trials idx."""
    def ok(m):
        return p1(f0_at(sw, trigger_times(P, m, shape))[idx]) >= F0_REQ
    lo, hi = -5.0, 40.0
    if not ok(hi):
        return float("nan")
    while hi - lo > 0.005:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return math.ceil(hi * 1000.0) / 1000.0          # rounded up, so the calibrated property still holds


def calibrate_switch(sw, idx):
    """Earliest common switch time on the 0.05 min grid with empirical 1st percentile of resolved F0 >= 8."""
    for i in range(sw["tab"].shape[0]):
        if p1(sw["tab"][i, idx]) >= F0_REQ:
            return round((sw["n0"] + i) * DT_FV, 2)
    return float("nan")


def cp_upper(k, n):
    """One-sided 95% Clopper-Pearson upper bound for a binomial proportion."""
    return 1.0 if k >= n else float(beta_dist.ppf(0.95, k + 1, n - k))


def archived_q_conf():
    rng = np.random.default_rng(1042)
    res = np.abs(rng.normal(0.0, 0.067, size=1000))
    return float(np.sort(res)[min(999, math.ceil((1000 + 1) * (1.0 - 0.001)) - 1)])


def main():
    global DATA, RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--datadir", default=str(DATA))
    ap.add_argument("--quick", action="store_true", help="60-trial smoke test, no solver checks")
    args = ap.parse_args()
    DATA = Path(args.datadir)
    RESULTS = DATA.parent / "results"
    out_dir = Path(args.outdir)
    t_start = time.time()

    trials = load_trials()
    if args.quick:
        trials = trials[:60]
    ntr = len(trials)
    tr = {"id": np.array([int(r["trial_id"]) for r in trials]),
          "alpha_mult": np.array([float(r["retort_alpha_mult"]) for r in trials]),
          "T0": np.array([T0_NOM + (float(r["retort_T0_C"]) - 38.0) * 0.65 for r in trials]),
          "z": np.array([float(r["retort_z_bot_C"]) for r in trials]),
          "dip": np.array([float(r["retort_dT_steam_dip_C"]) for r in trials])}
    tr["alpha"] = ALPHA_NOM * tr["alpha_mult"]
    folds = [np.where(tr["id"] % 2 == 1)[0], np.where(tr["id"] % 2 == 0)[0]]
    all_idx = np.arange(ntr)

    Bp, Bc = modal_basis(NR, NZ), modal_basis(NR_CTRL, NZ_CTRL)
    print(f"modal bases ready ({time.time() - t_start:.0f} s); deviation from the sparse operator {Bp['dev']:.1e}")

    # ---------------------------------------------------------------- solver checks (+ modal form, + archived reproduction)
    checks = [] if args.quick else solver_checks()
    if not args.quick:
        one = {"alpha": np.array([ALPHA_NOM]), "T0": np.array([T0_NOM]), "z": np.array([10.0]), "dip": np.array([0.0])}
        tg = np.arange(int(round(79.5 / DT_FV)) + 1) * DT_FV
        U, heat, t_h = retort_profile(tg, "crt", one["T0"], one["dip"], 121.1, 59.5)
        r = plant_run(Bp, one["alpha"], one["T0"], one["z"], U, heat, t_h + 20.0)
        checks.append({"grid_nr": f"{NR} (modal CN, vectorised)", "grid_nz": NZ, "dt_min": DT_FV,
                       "T_centre_10min": "", "T_centre_20min": "", "T_centre_30min": "", "T_centre_40min": "",
                       "F0_nominal_CRT_min": round(float(r["F0"][0]), 3),
                       "thiamine_retention_pct": round(float(r["CV"][0]), 3),
                       "surface_cook_C0_min": round(float(r["C0"][0]), 2)})

    # ---------------------------------------------------------------- policy trajectories
    q_conf = archived_q_conf()
    pol = []          # each: dict(name, label, info, evidence, kind, ... , full-grid U/heat/t_end, reduced-grid Tr/heat/t_end)

    def add_open(name, label, info, shape, T_set, t_sw, inclusive=True, tune=""):
        pol.append({"name": name, "label": label, "info": info, "shape": shape, "T_set": T_set,
                    "t_sw": np.broadcast_to(np.asarray(t_sw, float), (ntr,)).copy(), "inclusive": inclusive,
                    "dipf": 1.0, "pre": None, "tune": tune})

    add_open("CRT_121.1C_59.5min", "Fixed schedule, 121.1 degC for 59.5 min", "none (assigned schedule)", "crt", 121.1, 59.5)
    add_open("CRT_121.1C_68.0min", "Extended fixed schedule, 121.1 degC for 68.0 min", "none (assigned schedule)", "crt", 121.1, 68.0)
    add_open("VRT_assigned_59.8min", "Assigned VRT schedule (108 / 129.6 / 103 degC, heating 59.8 min)", "none (assigned schedule)",
             "vrt", VRT_SET, 48.3)
    for dipf, nm in ((0.22, "ADAPTIVE_archived_privileged_dip22pct"), (1.0, "ADAPTIVE_privileged_full_dip")):
        a = archived_adaptive(tr["alpha_mult"] * ALPHA_NOM, tr["T0"], tr["z"], tr["dip"], q_conf, dipf=dipf)
        pol.append({"name": nm, "label": "Adaptive surrogate, privileged information" +
                    (" (archived: dip attenuated to 22%)" if dipf < 1 else " (full dip)"),
                    "info": "trial diffusivity, initial temperature and z-value; dip known in advance; three-node state",
                    "shape": "vrt", "T_set": a["T_eff"], "t_sw": a["t_s"], "inclusive": False, "dipf": dipf,
                    "pre": np.where(a["t_s"] > 60.5, 60.5, np.inf), "tune": "", "red": a})
    print(f"archived adaptive replayed ({time.time() - t_start:.0f} s)")

    rng = np.random.default_rng(NOISE_SEED)
    n_noise = int(round(T_FORCE["crt"] / DT_DEC)) + 1
    noise = {"tr": rng.standard_normal((n_noise, 500))[:, :ntr], "it": rng.standard_normal(500)[:ntr],
             "pr": rng.standard_normal((n_noise, 500))[:, :ntr]}
    sw, sig = {}, {}
    T_shape = {"crt": 121.1, "vrt": VRT_SET}
    for shape in ("crt", "vrt"):
        sw[shape] = switch_table(Bp, tr, shape, T_shape[shape])
        sig[shape], _ = controller_signals(Bc, tr, shape, T_shape[shape], sw[shape], noise)
        print(f"switch table and controller signals, {shape} ({time.time() - t_start:.0f} s)")

    info_ff = "measured retort and initial temperature (SD 0.5 K); nominal diffusivity; z = 10 degC"
    info_pr = info_ff + "; sentinel-can centre probe (SD 0.5 K); diffusivity selected from a model bank"
    for shape, sh_lbl in (("crt", "CRT 121.1 degC"), ("vrt", "VRT 129.6 degC pulse")):
        for key, info, lbl in (("ff", info_ff, "feed-forward"), ("pr", info_pr, "probe")):
            P = sig[shape][key]
            m_in = calibrate_margin(P, sw[shape], shape, all_idx)
            m_cf = [calibrate_margin(P, sw[shape], shape, folds[1 - f]) for f in (0, 1)]
            ts_cf = np.empty(ntr)
            for f in (0, 1):
                ts_cf[folds[f]] = trigger_times(P, m_cf[f], shape)[folds[f]]
            cf_count = int((f0_at(sw[shape], ts_cf) < F0_REQ).sum())
            for m, tag in ((0.0, "m0"), (m_in, "tuned_P1")):
                p = {"name": f"ADAPTIVE_{key.upper()}_{shape.upper()}_{tag}",
                     "label": f"Adaptive, {lbl}, {sh_lbl}, margin m = {m:.3f} min", "info": info, "shape": shape,
                     "T_set": T_shape[shape], "t_sw": trigger_times(P, m, shape), "inclusive": True, "dipf": 1.0,
                     "pre": None, "tune": f"m = {m:.3f} min" if tag != "m0" else "m = 0",
                     "field": not (tag == "m0" and shape == "crt")}
                if tag != "m0":
                    p["crossfit"] = (f"m = {m_cf[0]:.3f} / {m_cf[1]:.3f} min (odd / even ids)", cf_count)
                pol.append(p)
            if key == "pr":
                Pb = sig[shape]["pr_bias"]
                pol.append({"name": f"ADAPTIVE_PR_{shape.upper()}_tuned_P1_probe_bias_+{PROBE_BIAS}K",
                            "label": f"Adaptive, probe, {sh_lbl}, margin m = {m_in:.3f} min, probe reads {PROBE_BIAS} K high",
                            "info": info_pr + f"; constant probe offset +{PROBE_BIAS} K (sensitivity)", "shape": shape,
                            "T_set": T_shape[shape], "t_sw": trigger_times(Pb, m_in, shape), "inclusive": True,
                            "dipf": 1.0, "pre": None, "tune": f"m = {m_in:.3f} min (calibrated without offset)",
                            "field": False})
        ts_in = calibrate_switch(sw[shape], all_idx)
        ts_cf = [calibrate_switch(sw[shape], folds[1 - f]) for f in (0, 1)]
        tsv = np.empty(ntr)
        for f in (0, 1):
            tsv[folds[f]] = ts_cf[f]
        cf_count = int((f0_at(sw[shape], tsv) < F0_REQ).sum())
        if shape == "crt":
            add_open(f"CRT_121.1C_tuned_P1_{ts_in:.2f}min", f"Fixed CRT schedule, 121.1 degC for {ts_in:.2f} min (tuned)",
                     "none (assigned schedule)", "crt", 121.1, ts_in, tune=f"hold = {ts_in:.2f} min")
        else:
            add_open(f"VRT_tuned_P1_pulse_to_{ts_in:.2f}min", f"VRT schedule, 129.6 degC pulse to {ts_in:.2f} min (tuned)",
                     "none (assigned schedule)", "vrt", VRT_SET, ts_in, tune=f"pulse end = {ts_in:.2f} min")
        pol[-1]["crossfit"] = (f"{ts_cf[0]:.2f} / {ts_cf[1]:.2f} min (odd / even ids)", cf_count)
    print(f"margins calibrated ({time.time() - t_start:.0f} s)")

    # ---------------------------------------------------------------- evaluate every policy on both models
    archived = {}
    arch_map = {"1_CRT_Optimal_121.1C_Fixed_Schedule": "CRT_121.1C_59.5min", "2_CRT_Overcooked_Industrial_Safety_Margin": "CRT_121.1C_68.0min",
                "3_Deterministic_VRT_Pareto_Nominal_NMPC": "VRT_assigned_59.8min",
                "4_Conformal_Robust_VRT_Twin_UKF_SH_NMPC": "ADAPTIVE_archived_privileged_dip22pct"}
    if (RESULTS / "Ch10_Table_10_4_Scenario_IA_Retort_Benchmark.csv").exists():
        with open(RESULTS / "Ch10_Table_10_4_Scenario_IA_Retort_Benchmark.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["control_paradigm"] in arch_map:
                    archived[arch_map[r["control_paradigm"]]] = (
                        f"F0 {r['F0_mean_min']} +- {r['F0_sd_min']}; <8: {r['under_sterilised_batches_count']}; "
                        f"C_V {r['thiamine_retention_mean_pct']}; C0 {r['surface_cook_C0_mean_min']}; "
                        f"cycle {r['batch_cycle_time_mean_min']}; exergy {r['specific_steam_exergy_kJ_per_can']}")
    trial_rows, summary = [], []
    for p in pol:
        Tset = np.broadcast_to(np.asarray(p["T_set"], float), (ntr,))
        t_end = (p["t_sw"] if p["shape"] == "crt" else p["t_sw"] + 11.5) + 20.0
        nf = int(round(t_end.max() / DT_FV))
        U, heat, _ = retort_profile(np.arange(nf + 1) * DT_FV, p["shape"], tr["T0"], tr["dip"], Tset, p["t_sw"],
                                    p["dipf"], p["inclusive"], p["pre"])
        if p.get("field", True):
            full = plant_run(Bp, tr["alpha"], tr["T0"], tr["z"], U, heat, t_end)
        else:                                       # F0 only, exact from the switch table; no field run
            nan = np.full(ntr, np.nan)
            full = {"F0": f0_at(sw[p["shape"]], p["t_sw"]), "CV": nan, "C0": nan, "ex": nan}
        if "red" in p:
            red = p["red"]
        else:
            n4 = int(np.ceil(t_end.max() / 0.25 - 1e-9))
            U4, h4, _ = retort_profile(np.arange(n4 + 1) * 0.25, p["shape"], tr["T0"], tr["dip"], Tset, p["t_sw"],
                                       p["dipf"], p["inclusive"], p["pre"])
            F, C, K, X = reduced_batch(tr["alpha"], tr["T0"], tr["z"], U4, h4, t_end)
            red = {"F0": F, "CV": C, "C0": K, "ex": X, "t_h": t_end - 20.0}
        # the reduced model's cycle ends on its 0.25 min grid; the resolved model's at t_end
        cyc_red = np.ceil((red["t_h"] + 20.0) / 0.25 - 1e-9) * 0.25 if "red" not in p else red["t_h"] + 20.0
        for i in range(ntr):
            trial_rows.append({"schedule": p["name"], "trial_id": int(tr["id"][i]),
                               "F0_reduced_min": red["F0"][i], "thiamine_reduced_pct": red["CV"][i],
                               "surface_cook_reduced_min": red["C0"][i],
                               "F0_full_min": full["F0"][i], "thiamine_full_pct": full["CV"][i],
                               "surface_cook_full_min": full["C0"][i],
                               "switch_or_hold_end_min": round(float(p["t_sw"][i]), 2),
                               "heating_end_min": round(float(t_end[i] - 20.0), 2), "cycle_time_min": round(float(t_end[i]), 2),
                               "exergy_reduced_kJ_per_can": red["ex"][i], "exergy_full_kJ_per_can": full["ex"][i]})
        f_r, f_f = red["F0"], full["F0"]
        R = lambda v, d: "" if not np.isfinite(v) else round(float(v), d)
        rel = (f_r - f_f) / f_f * 100.0
        k_r, k_f = int((f_r < F0_REQ).sum()), int((f_f < F0_REQ).sum())
        row = {"schedule": p["name"], "n_trials": ntr,
               "F0_reduced_mean_min": round(f_r.mean(), 2), "F0_full_mean_min": round(f_f.mean(), 2),
               "F0_relative_error_mean_pct": round(rel.mean(), 1), "F0_relative_error_sd_pct": round(rel.std(), 1),
               "F0_relative_error_min_pct": round(rel.min(), 1), "F0_relative_error_max_pct": round(rel.max(), 1),
               "under_target_reduced_count": k_r, "under_target_full_count": k_f,
               "classification_disagreements": int(((f_r < F0_REQ) != (f_f < F0_REQ)).sum()),
               "thiamine_reduced_mean_pct": round(red["CV"].mean(), 1), "thiamine_full_mean_pct": R(full["CV"].mean(), 1),
               "thiamine_abs_error_mean_pct_points": R((red["CV"] - full["CV"]).mean(), 2),
               "surface_cook_reduced_mean_min": round(red["C0"].mean(), 1), "surface_cook_full_mean_min": R(full["C0"].mean(), 1),
               "surface_cook_relative_error_mean_pct": R(((red["C0"] - full["C0"]) / full["C0"] * 100.0).mean(), 1),
               "evidence_status": "model_to_model_verification_of_reduction_and_executed_benchmark_on_resolved_conduction_not_experimental_validation",
               "policy": p["label"], "information_used": p["info"], "tuning": p["tune"],
               "F0_reduced_sd_min": round(f_r.std(), 2), "F0_full_sd_min": round(f_f.std(), 2),
               "F0_reduced_p0_1_min": round(float(np.percentile(f_r, 0.1)), 2), "F0_reduced_p1_min": round(p1(f_r), 2),
               "F0_full_p0_1_min": round(float(np.percentile(f_f, 0.1)), 2), "F0_full_p1_min": round(p1(f_f), 2),
               "F0_full_p5_min": round(float(np.percentile(f_f, 5.0)), 2), "F0_full_median_min": round(float(np.median(f_f)), 2),
               "under_target_reduced_upper95_pct": round(100 * cp_upper(k_r, ntr), 2),
               "under_target_full_pct": round(100 * k_f / ntr, 2), "under_target_full_upper95_pct": round(100 * cp_upper(k_f, ntr), 2),
               "F0_shortfall_at_full_p1_min": round(max(0.0, F0_REQ - p1(f_f)), 2),
               "thiamine_reduced_sd_pct": round(red["CV"].std(), 1), "thiamine_full_sd_pct": R(full["CV"].std(), 1),
               "surface_cook_reduced_sd_min": round(red["C0"].std(), 1), "surface_cook_full_sd_min": R(full["C0"].std(), 1),
               "cycle_time_full_mean_min": round(t_end.mean(), 1), "cycle_time_full_sd_min": round(t_end.std(), 1),
               "cycle_time_reduced_mean_min": round(float(np.mean(cyc_red)), 1),
               "exergy_reduced_mean_kJ_per_can": round(red["ex"].mean(), 1), "exergy_full_mean_kJ_per_can": R(full["ex"].mean(), 1),
               "energy_full_MJ_per_kg": R(full["ex"].mean() / 1000.0 / (0.2105 * 0.45), 2),
               "crossfit_tuning": p.get("crossfit", ("", ""))[0], "crossfit_under_target_full_count": p.get("crossfit", ("", ""))[1],
               "archived_runner_values": archived.get(p["name"], "")}
        summary.append(row)
        print(f"{p['name']:<52s} red {row['F0_reduced_mean_min']:6.2f} ({k_r:3d})  full {row['F0_full_mean_min']:6.2f} "
              f"({k_f:3d}, P1 {row['F0_full_p1_min']:5.2f})  CV {row['thiamine_full_mean_pct']!s:>5}  "
              f"cycle {row['cycle_time_full_mean_min']:5.1f}  ({time.time() - t_start:.0f} s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    if checks:
        with open(out_dir / "Ch10_Reduction_Error_Solver_Checks.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(checks[0].keys()))
            w.writeheader()
            w.writerows(checks)
    for name, rows in (("Ch10_Reduction_Error_Trials.csv", trial_rows), ("Ch10_Reduction_Error_Summary.csv", summary)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
