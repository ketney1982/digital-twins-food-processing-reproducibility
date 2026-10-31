"""Physical-data check of the bread-baking scenario (Chapter 11, Scenario IIA-2).

Data: measured temperatures of a leavened bread baked at 203 degC for 30 min (Zanoni, Peri & Pierucci, J. Food Eng. 19 (1993) 389-398,
Fig. 1; library record 584): oven, upper crust, lower crust, crumb 1 cm below the upper surface and crumb centre. The paper gives
the curves only as a raster figure, so the values in data/external/bread_baking_temperatures_Zanoni1993.csv were read from an enlarged
rendering of the figure at 5 min intervals (reading uncertainty about +-3 degC). Conditions (Table 1 of the paper): 341 g, radius 5.5 cm,
initial height 4.6 cm, initial temperature 26 degC, dough moisture 44 % (wet basis), air speed 1.4 m/s; relative height
H/H0 = 1.07 + 9.2e-4 t (t <= 1140 s), 2.61 - 4.5e-4 t afterwards (the paper's regression of the measured volume).

Model: one-dimensional conduction through the loaf height (heated from the top and, through the mould bottom, from below) in material
coordinates with the measured expansion, an effective conductivity K (standing for conduction plus evaporation-condensation, as in
Purlis, J. Food Eng. 103 (2011) 92-102, record 422) and a latent-heat jump at 99-101 degC for an assumed evaporable water fraction.
Fitted to the four measured series: K and the heat-transfer coefficients of the top and the bottom. For comparison the same fit is
repeated with K fixed at the conduction-only value used in verify_ch11_baking_resolved.py (0.35 W/(m K)).

Scenario transfer: the calibrated K (and h as fitted, 12 and 25 W/(m2 K)) is applied to the loaf of the scenario (half-thickness 42 mm,
no expansion) under the two oven programmes supplied for this work (artisan: 240 degC for 15 min then 210 degC; wholemeal: 165 degC) and
under the 203 degC of the experiment. Starch gelatinisation uses the first-order Arrhenius kinetics of Purlis (2011): K0 = 2.8e18 1/s,
Ea = 139 kJ/mol; the loaf is taken as baked when the coldest point reaches alpha >= 0.98 (same criterion as Purlis 2011).

Scope: one loaf geometry, one oven, temperatures read from a figure; the transfer to the scenario loaf and oven is an extrapolation.

Reconstructed (geometrically compatible) bread scenario, added after the transfer above showed that the 42 mm half-thickness dough
piece cannot gelatinise in 18.2-22 min: the bread product of Scenario IIA-2 is redefined as the measured loaf itself (340 g dough
piece, radius 55 mm, dough height 46 mm = half-height 23 mm, rising with the measured expansion law to about 97 mm), baked on the
artisan programme (240 degC for 15 min, then 210 degC) with a residence time of 30 min (the measured expansion law is used over its
0-30 min range and the height is held constant afterwards). Recalibration uncertainty: the fit is repeated
one factor at a time with the read temperatures shifted by -3 and +3 degC, the evaporable water fraction set to 0.15 and 0.35 and the
specific heat set to 2600 and 3400 J/(kg K); every refitted set (K, h_top, h_bottom) is carried to the reconstructed loaf. The 250
oven heat-transfer multipliers of data/monte_carlo_disturbances_seed42.csv (column baking_H_oven_mult) scale both coefficients; the
coldest-point gelatinisation degree at 30 min is reported for the nominal set and for the slowest and fastest refitted sets. A lower
convective coefficient (25 and 12 W/(m2 K) on both faces) is reported as a separate sensitivity because a tunnel oven need not
reproduce the heat transfer of the laboratory oven. Only the fixed programme is computed: the design slots of the scenario define no
oven programme, so no slot-specific gelatinisation can be derived.
Output: Ch11_Baking_Physical_Check.csv.   Run: python verify_ch11_baking_physical.py
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "external" / "bread_baking_temperatures_Zanoni1993.csv"
RHO0, H0, CP, T0 = 780.0, 0.046, 3000.0, 26.0       # 341 g in pi 5.5^2 x 4.6 cm3 gives about 780 kg/m3
W_WET, EVAP_FRAC, LV = 0.44, 0.25, 2.26e6            # measured dough moisture; assumed evaporable fraction (bake loss about 11 %)
K0_GEL, EA_GEL, R_GAS = 2.8e18, 139000.0, 8.314462618
K_CONDUCTION_ONLY = 0.35


def read_data():
    with open(DATA, newline="", encoding="utf-8") as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return rows


def rel_height(t_s):
    return 1.07 + 9.2e-4 * t_s if t_s <= 1140.0 else 2.61 - 4.5e-4 * t_s


def simulate(K, h_top, h_bot, t_end_min, T_air, height, mass_area, n=40, dt=2.0, probes=(), T_init=T0, track_alpha=False):
    """Generic slab in material coordinates; height(t) gives the thickness (m); T_air(t_min) the oven programme.
    Returns probe histories at 1 min intervals, centre history and (optionally) the coldest-point gelatinisation degree."""
    dxi = 1.0 / n
    latent = LV * W_WET * EVAP_FRAC * mass_area * dxi                  # J/m2 released per cell over 99-101 degC
    T = np.full(n, T_init)
    alpha = np.zeros(n)
    out = {"t": [], "probe": [], "centre": [], "alpha_min": [], "T_min": []}
    steps = int(round(t_end_min * 60 / dt))
    for s in range(steps):
        t = s * dt
        H = height(t)
        cap = mass_area * CP * dxi
        band = (T > 99.0) & (T < 101.0)
        c_cell = cap + np.where(band, latent / 2.0, 0.0)             # J/(m2 K) per cell
        a = K / (H * dxi)
        gt = 1.0 / (1.0 / h_top + H * dxi / (2.0 * K))
        gb = 1.0 / (1.0 / h_bot + H * dxi / (2.0 * K))
        diag = np.full(n, 2 * a); lower = np.full(n, -a); upper = np.full(n, -a)
        diag[0] = a + gt; diag[-1] = a + gb
        rhs = c_cell / dt * T
        rhs[0] += gt * T_air(t / 60.0); rhs[-1] += gb * T_air(t / 60.0)
        ab = np.vstack([np.r_[0.0, upper[:-1]], diag + c_cell / dt, np.r_[lower[1:], 0.0]])
        T = solve_banded((1, 1), ab, rhs)
        if track_alpha:
            k = K0_GEL * np.exp(-EA_GEL / (R_GAS * (T + 273.15)))
            alpha = 1.0 - (1.0 - alpha) * np.exp(-k * dt)
        if (s + 1) % int(60 / dt) == 0:
            out["t"].append((s + 1) * dt / 60.0)
            out["probe"].append([float(np.interp(p, (np.arange(n) + 0.5) * dxi, T)) for p in probes])
            out["centre"].append(float(np.interp(0.5, (np.arange(n) + 0.5) * dxi, T)))
            out["alpha_min"].append(float(alpha.min())); out["T_min"].append(float(T.min()))
    return out


def fit(data, free_K=True):
    obs = {k: np.array([r[k] for r in data]) for k in ("upper_crust_C", "lower_crust_C", "crumb_1cm_C", "crumb_centre_C")}
    tmin = np.array([r["time_min"] for r in data])
    oven = lambda t_min: float(np.interp(t_min, tmin, [r["oven_C"] for r in data]))
    height = lambda t: H0 * rel_height(t)
    mass_area = RHO0 * H0
    probes = (0.0, 0.217, 0.5, 1.0)                                  # upper crust, 1 cm crumb, centre, lower crust

    def run(p):
        K = math.exp(p[0]) if free_K else K_CONDUCTION_ONLY
        ht, hb = math.exp(p[1]), math.exp(p[2])
        r = simulate(K, ht, hb, 30.0, oven, height, mass_area, probes=probes)
        idx = [int(round(m)) - 1 for m in tmin[1:]]
        P = np.array(r["probe"])
        return np.stack([P[idx, 0], P[idx, 3], P[idx, 1], P[idx, 2]], axis=1)       # same order as obs keys

    def res(p):
        sim = run(p)
        return np.concatenate([sim[:, j] - obs[k][1:] for j, k in enumerate(("upper_crust_C", "lower_crust_C", "crumb_1cm_C", "crumb_centre_C"))])
    sol = least_squares(res, [math.log(0.5), math.log(25.0), math.log(10.0)], x_scale=1.0, max_nfev=40)
    sim = run(sol.x)
    rmse = {k: float(np.sqrt(np.mean((sim[:, j] - obs[k][1:]) ** 2))) for j, k in enumerate(obs)}
    return (math.exp(sol.x[0]) if free_K else K_CONDUCTION_ONLY, math.exp(sol.x[1]), math.exp(sol.x[2])), rmse


def measured_alpha(data, column, shift=0.0):
    """Gelatinisation degree integrated along a measured temperature history (linear interpolation, 6 s steps)."""
    t = np.array([r["time_min"] for r in data]); T = np.array([r[column] for r in data]) + shift
    tt = np.arange(0.0, 30.001, 0.1); Ti = np.interp(tt, t, T)
    a, t98, hist = 0.0, None, []
    for i in range(len(tt)):
        k = K0_GEL * math.exp(-EA_GEL / (R_GAS * (Ti[i] + 273.15)))
        a = 1.0 - (1.0 - a) * math.exp(-k * 6.0)
        hist.append(a)
        if t98 is None and a >= 0.98:
            t98 = float(tt[i])
    return t98, float(np.interp(22.0, tt, hist))


def model_experiment(K, ht, hb, data):
    tmin = np.array([r["time_min"] for r in data])
    oven = lambda t_min: float(np.interp(t_min, tmin, [r["oven_C"] for r in data]))
    r = simulate(K, ht, hb, 30.0, oven, lambda t: H0 * rel_height(t), RHO0 * H0, track_alpha=True)
    t = np.array(r["t"]); amin = np.array(r["alpha_min"])
    return next((float(x) for x, a in zip(t, amin) if a >= 0.98), None), float(r["centre"][-1])


def equivalent_thickness(K, h, programme, target_min, label, rows):
    """Total slab thickness (no expansion) for which the coldest point reaches alpha >= 0.98 at target_min."""
    def t98(H_total):
        r = simulate(K, h, h, 60.0, programme, lambda t: H_total, RHO0 * H_total, track_alpha=True)
        return next((float(x) for x, a in zip(r["t"], r["alpha_min"]) if a >= 0.98), 99.0)
    lo, hi = 0.008, 0.084
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t98(mid) < target_min else (lo, mid)
    rows.append({"case": label, "K_W_mK": round(K, 2), "h_W_m2K": round(h, 1), "target_time_to_alpha_0.98_min": target_min,
                 "equivalent_total_thickness_mm": round(1000 * 0.5 * (lo + hi), 1)})


def scenario(K, h, programme, label, rows):
    height = lambda t: 0.084
    r = simulate(K, h, h, 120.0, programme, height, RHO0 * 0.084, track_alpha=True)
    t = np.array(r["t"]); amin = np.array(r["alpha_min"]); centre = np.array(r["centre"])
    t98 = next((float(x) for x, a in zip(t, amin) if a >= 0.98), None)
    t96 = next((float(x) for x, c in zip(t, centre) if c >= 96.0), None)
    i22 = int(np.argmin(abs(t - 22.0)))
    rows.append({"case": label, "K_W_mK": round(K, 2), "h_W_m2K": round(h, 1),
                 "core_T_at_22min_C": round(centre[i22], 1), "alpha_coldest_at_22min": round(amin[i22], 3),
                 "time_to_core_96C_min": t96 if t96 is not None else ">120", "time_to_alpha_0.98_min": t98 if t98 is not None else ">120",
                 "reduced_model_alpha_at_18p2min": 0.981})


MC_FILE = HERE.parent / "data" / "monte_carlo_disturbances_seed42.csv"
ARTISAN = lambda t: 240.0 if t < 15.0 else 210.0
RESIDENCE_MIN = 30.0
LOAF_H0 = H0                                                          # reconstructed loaf = measured loaf (46 mm dough height)


def with_props(evap, cp, fn, *args):
    """Run fn with the evaporable water fraction and specific heat temporarily replaced (module constants used by simulate)."""
    global EVAP_FRAC, CP
    old = (EVAP_FRAC, CP)
    EVAP_FRAC, CP = evap, cp
    try:
        return fn(*args)
    finally:
        EVAP_FRAC, CP = old


def loaf_run(K, ht, hb, programme, t_end):
    # measured expansion law used within its range (0-30 min); the height is held at its 30 min value afterwards
    return simulate(K, ht, hb, t_end, programme, lambda t: LOAF_H0 * rel_height(min(t, 1800.0)), RHO0 * LOAF_H0, track_alpha=True)


def loaf_t98(K, ht, hb, programme, t_end=80.0):
    r = loaf_run(K, ht, hb, programme, t_end)
    return next((float(x) for x, a in zip(r["t"], r["alpha_min"]) if a >= 0.98), None)


def loaf_alpha_at(K, ht, hb, programme, t_min):
    return loaf_run(K, ht, hb, programme, t_min)["alpha_min"][-1]


def compatible_scenario(data, rows):
    """Recalibration ensemble, reconstructed loaf and the 250 heat-transfer multipliers (see module docstring)."""
    variants = [("nominal", 0.0, 0.25, 3000.0), ("reading_minus3C", -3.0, 0.25, 3000.0), ("reading_plus3C", 3.0, 0.25, 3000.0),
                ("evaporable_0.15", 0.0, 0.15, 3000.0), ("evaporable_0.35", 0.0, 0.35, 3000.0),
                ("cp_2600", 0.0, 0.25, 2600.0), ("cp_3400", 0.0, 0.25, 3400.0)]
    progs = {"artisan_240C_15min_then_210C": ARTISAN, "wholemeal_165C": lambda t: 165.0, "experiment_203C": lambda t: 203.0}
    fitted = {}
    for name, shift, evap, cp in variants:
        d = [{k: (v + shift if k not in ("time_min", "oven_C") else v) for k, v in r.items()} for r in data]
        (K, ht, hb), rm = with_props(evap, cp, fit, d, True)
        t98 = {p: with_props(evap, cp, loaf_t98, K, ht, hb, f) for p, f in progs.items()}
        a30 = with_props(evap, cp, loaf_alpha_at, K, ht, hb, ARTISAN, RESIDENCE_MIN)
        fitted[name] = (K, ht, hb, evap, cp, t98["artisan_240C_15min_then_210C"])
        rows.append({"case": "reconstructed_loaf_recalibration_" + name, "K_W_mK": round(K, 2), "h_W_m2K": f"top {ht:.1f} / bottom {hb:.1f}",
                     "evaporable_fraction": evap, "cp_J_kgK": cp, "reading_shift_C": shift,
                     **{f"rmse_{k}": round(v, 1) for k, v in rm.items()},
                     "t98_reconstructed_loaf_artisan_min": t98["artisan_240C_15min_then_210C"], "t98_reconstructed_loaf_wholemeal_min": t98["wholemeal_165C"],
                     "t98_reconstructed_loaf_203C_min": t98["experiment_203C"], "alpha_coldest_at_30min_artisan": round(a30, 4)})
        print(rows[-1], flush=True)
    Kn, htn, hbn, *_ = fitted["nominal"]
    for h in (25.0, 12.0):
        rows.append({"case": f"reconstructed_loaf_low_heat_transfer_h_{h:.0f}", "K_W_mK": round(Kn, 2), "h_W_m2K": f"top {h:.1f} / bottom {h:.1f}",
                     "t98_reconstructed_loaf_artisan_min": loaf_t98(Kn, h, h, ARTISAN, 120.0),
                     "alpha_coldest_at_30min_artisan": round(loaf_alpha_at(Kn, h, h, ARTISAN, RESIDENCE_MIN), 4)})
        print(rows[-1], flush=True)
    with open(MC_FILE, newline="", encoding="utf-8") as fh:
        mult = np.array([float(r["baking_H_oven_mult"]) for r in list(csv.DictReader(fh))[:250]])
    slow = max(fitted, key=lambda k: fitted[k][5]); fast = min(fitted, key=lambda k: fitted[k][5])
    for label in dict.fromkeys(("nominal", slow, fast)):
        K, ht, hb, evap, cp, _ = fitted[label]
        a = np.array([with_props(evap, cp, loaf_alpha_at, K, ht * m, hb * m, ARTISAN, RESIDENCE_MIN) for m in mult])
        t98_low = with_props(evap, cp, loaf_t98, K, ht * mult.min(), hb * mult.min(), ARTISAN)
        rows.append({"case": "reconstructed_loaf_250_trials_" + label, "K_W_mK": round(K, 2), "h_W_m2K": f"top {ht:.1f} / bottom {hb:.1f} x multiplier",
                     "h_multiplier_min_mean_max": f"{mult.min():.3f}/{mult.mean():.3f}/{mult.max():.3f}", "h_multiplier_sd": round(float(mult.std(ddof=1)), 4),
                     "alpha_coldest_at_30min_mean_pct": round(100 * float(a.mean()), 2), "alpha_coldest_at_30min_sd_pct": round(100 * float(a.std(ddof=1)), 2),
                     "alpha_coldest_at_30min_min_pct": round(100 * float(a.min()), 2), "trials_below_0.98": int((a < 0.98).sum()),
                     "t98_at_lowest_multiplier_min": t98_low})
        print(rows[-1], flush=True)


def main():
    data = read_data()
    (K, ht, hb), rm_free = fit(data, True)
    (_, ht0, hb0), rm_fix = fit(data, False)
    rows = [{"case": "fit_to_Zanoni_1993_effective_K", "K_W_mK": round(K, 2), "h_W_m2K": f"top {ht:.1f} / bottom {hb:.1f}",
             **{f"rmse_{k}": round(v, 1) for k, v in rm_free.items()}},
            {"case": "fit_to_Zanoni_1993_conduction_only_K", "K_W_mK": K_CONDUCTION_ONLY, "h_W_m2K": f"top {ht0:.1f} / bottom {hb0:.1f}",
             **{f"rmse_{k}": round(v, 1) for k, v in rm_fix.items()}}]
    t98_meas, a22_meas = measured_alpha(data, "crumb_centre_C")
    t98_lo = measured_alpha(data, "crumb_centre_C", -3.0)[0]; t98_hi = measured_alpha(data, "crumb_centre_C", 3.0)[0]
    t98_1cm = measured_alpha(data, "crumb_1cm_C")[0]
    t98_mod, tc_mod = model_experiment(K, ht, hb, data)
    rows.append({"case": "measured_gelatinisation_203C_experiment", "time_to_alpha_0.98_min": t98_meas,
                 "alpha_coldest_at_22min": round(a22_meas, 3), "time_to_alpha_0.98_min_reading_minus3C": t98_lo,
                 "time_to_alpha_0.98_min_reading_plus3C": t98_hi, "time_to_alpha_0.98_1cm_layer_min": t98_1cm})
    rows.append({"case": "model_on_the_experiment_203C", "time_to_alpha_0.98_min": t98_mod if t98_mod is not None else ">30 (centre at 30 min %.0f C)" % tc_mod,
                 "K_W_mK": round(K, 2)})
    print(rows, flush=True)
    sc = []
    progs = {"artisan_240C_15min_then_210C": lambda t: 240.0 if t < 15.0 else 210.0, "wholemeal_165C": lambda t: 165.0, "experiment_203C": lambda t: 203.0}
    for name, prog in progs.items():
        for h in (ht, 12.0, 25.0):
            scenario(K, h, prog, name, sc)
            print(sc[-1], flush=True)
    for name, prog in progs.items():
        for target in (18.2, 22.0):
            equivalent_thickness(K, ht, prog, target, "equivalent_thickness_" + name, sc)
            print(sc[-1], flush=True)
    rc = []
    compatible_scenario(data, rc)
    keys = sorted({k for r in rows + sc + rc for k in r})
    with open(HERE / "Ch11_Baking_Physical_Check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["evidence_status"]); w.writeheader()
        for r in rows + sc:
            r["evidence_status"] = "physical_temperatures_read_from_a_published_figure_one_loaf_one_oven_scenario_transfer_is_an_extrapolation"
            w.writerow(r)
        for r in rc:
            r["evidence_status"] = "executed_benchmark_calibrated_on_one_published_loaf_programme_transfer_is_an_extrapolation_not_validation"
            w.writerow(r)


if __name__ == "__main__":
    main()
