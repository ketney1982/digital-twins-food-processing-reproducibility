"""Reduction-error checks for the UHT (Ch10 IB), drying (Ch11 IIA-1) and fermentation (Ch12 IIIA) scenarios.

Each scenario in run_all_benchmarks.py is a reduced model. This script re-implements the reduced model
unchanged and compares it with a resolved counterpart built only from declared equations and declared parameters:

  UHT        event-driven 48 h campaign (production, fouling to the critical resistance, CIP, partial last
             cycle) instead of floor-count accounting, plus a sensitivity sweep over a dimensionless
             thermal-feedback gain g on the fouling rate, dRf/dt = k (1 + g Rf/Rf_max); g = 0 is the reduced model.
             Scenario IB is an assigned scenario (assigned fouling rates, stabilisation fractions and cleaning
             durations); this part is an accounting check of its campaign bookkeeping and an assigned sensitivity,
             not a resolved exchanger model, not a model-to-model verification of exchanger physics and not a validation.
  Drying     one-dimensional Fickian slab with the declared diffusivity descriptors (D_eff0 PD-080, Ea_diff PD-081) and GAB
             surface equilibrium (Xm, C, K: PD-082..PD-084; PD-085 is SEC_base_drying), solved by the analytical series, with the half-thickness
             calibrated so that the nominal case reproduces the reduced model's nominal core moisture;
             compared over the 250 Monte Carlo trials, for three assumed air humidities (the calibration absorbs the air temperature).
  Fermentation the same fed-batch equations integrated with a refined time step (1 h, 0.25 h, 0.05 h).

Scope: model-to-model checks of the reductions. They are not validation against plant data. Quantities that the
runner assigns as functions of disturbance magnitude (observer errors, interval coverage) are not computed
dynamics and are excluded.

Outputs (next to this script): Ch10_UHT_Reduction_Check.csv, Ch11_Drying_Reduction_Check.csv,
Ch12_Fermentation_Reduction_Check.csv.      Run: python verify_ch10_ch11_ch12_reduction_error.py
"""
import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results"


def load_params():
    out = {}
    with open(DATA / "kinetic_and_transport_parameters.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["process_id"], r["symbol"])] = float(r["value"])
    return out


def load_trials():
    with open(DATA / "monte_carlo_disturbances_seed42.csv", newline="", encoding="utf-8") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def write(name, rows):
    with open(HERE / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# =============================================================================== UHT (Chapter 10, Scenario IB)
UHT_STRATEGIES = [("1_Conventional_Fixed_Setpoint_UHT_82C_Prehold", 0.25, 1.50),
                  ("2_Open_Loop_PreHold_Optimised_UHT_91C_Grijspeerdt", 0.68, 1.50),
                  ("3_Closed_Loop_Fouling_Aware_ENMPC_Ultrasonic_MHE_CIP_Twin", 0.76, 1.075)]


def uht_indicators(n_cip, milk_t, beta, tau):
    steam = (292.0 - 25.0 * beta) + n_cip * 980.0 / milk_t
    effl = n_cip * (2380.0 * (tau / 1.50)) / milk_t
    return steam, effl


def uht_reduced(t_run, beta, tau, H=48.0, mdot=10.0):
    n = max(1, int(math.floor(H / (t_run + tau))))
    milk = (H - n * tau) * mdot
    return n, milk, *uht_indicators(n, milk, beta, tau)


def uht_resolved(t_run, beta, tau, H=48.0, mdot=10.0):
    """Event-driven campaign: production until the critical fouling resistance, then CIP; the last cycle is partial."""
    t, prod, n = 0.0, 0.0, 0
    while t < H - 1e-12:
        run = min(t_run, H - t)
        prod += run
        t += run
        if t >= H - 1e-12:
            break
        n += 1                                   # critical resistance reached before the end: CIP starts
        t += tau
    milk = prod * mdot
    return n, milk, *uht_indicators(n, milk, beta, tau)


def uht_check(trials):
    rows = []
    Rf_max = 0.00050
    for name, beta, tau in UHT_STRATEGIES:
        for g in (0.0, 0.5, 1.0, 2.0):
            acc = {k: [] for k in ("run", "n_red", "n_res", "milk_red", "milk_res", "st_red", "st_res", "ef_red", "ef_res")}
            for tr in trials[:200]:
                k = 0.000081 * (1.0 - 0.68 * beta) * tr["uht_k_foul_mult"] * (tr["uht_protein_load_mult"] ** 1.25)
                t_red = Rf_max / k
                t_res = t_red if g == 0.0 else (t_red / g) * math.log(1.0 + g)   # analytical time to Rf_max with feedback
                n1, m1, s1, e1 = uht_reduced(t_red, beta, tau)
                n2, m2, s2, e2 = uht_resolved(t_res, beta, tau)
                for key, v in zip(("run", "n_red", "n_res", "milk_red", "milk_res", "st_red", "st_res", "ef_red", "ef_res"),
                                  (t_res / t_red, n1, n2, m1, m2, s1, s2, e1, e2)):
                    acc[key].append(v)
            m = {k: float(np.mean(v)) for k, v in acc.items()}
            rows.append({"strategy": name, "feedback_gain_g": g,
                         "run_length_ratio_resolved_to_reduced": round(m["run"], 3),
                         "CIP_cycles_reduced": round(m["n_red"], 2), "CIP_cycles_resolved": round(m["n_res"], 2),
                         "milk_t_reduced": round(m["milk_red"], 1), "milk_t_resolved": round(m["milk_res"], 1),
                         "milk_error_pct": round((m["milk_res"] / m["milk_red"] - 1) * 100, 2),
                         "steam_MJ_t_reduced": round(m["st_red"], 1), "steam_MJ_t_resolved": round(m["st_res"], 1),
                         "effluent_L_t_reduced": round(m["ef_red"], 1), "effluent_L_t_resolved": round(m["ef_res"], 1),
                         "evidence_status": "assigned_scenario_accounting_check_g_is_an_assigned_sensitivity_not_validation"})
    return rows


# =============================================================================== Drying (Chapter 11, Scenario IIA-1)
R_GAS = 8.314462618


def gab_aw(X, Xm, C, K):
    a = C - 1.0
    b = (Xm * C / max(X, 1e-6)) + 2.0 - C
    disc = max(0.0, b * b + 4.0 * a)
    y = (-b + math.sqrt(disc)) / (2.0 * a)
    return float(min(0.995, max(0.005, y / K)))


def gab_x(aw, Xm, C, K):
    return Xm * C * K * aw / ((1 - K * aw) * (1 - K * aw + C * K * aw))


def reduced_drying(X0, dm, rh_amb):
    k = 0.02015 * (dm ** 0.38) * (1.0 - 0.0018 * (rh_amb - 55.0))     # tier 1: no gains
    return X0 * math.exp(-k * 190.0)


def slab_series(X0, Xeq, D, L, t_s, n_terms=60):
    n = np.arange(n_terms)
    tau = (2 * n + 1) ** 2 * math.pi ** 2 * D * t_s / (4 * L ** 2)
    core = Xeq + (X0 - Xeq) * float(np.sum((4 / math.pi) * (-1.0) ** n / (2 * n + 1) * np.exp(-tau)))
    mean = Xeq + (X0 - Xeq) * float(np.sum(8 / math.pi ** 2 / (2 * n + 1) ** 2 * np.exp(-tau)))
    return core, mean


def drying_check(P, trials):
    D0, Ea = P[("Scenario_IIA1_Drying", "D_eff0_apple")], P[("Scenario_IIA1_Drying", "Ea_diff_apple")]
    Xm, C, K = P[("Scenario_IIA1_Drying", "Xm_gab")], P[("Scenario_IIA1_Drying", "C_gab")], P[("Scenario_IIA1_Drying", "K_gab")]
    aw_max = P[("Scenario_IIA1_Drying", "aw_max_apple")]
    t_s = 190.0 * 60.0
    x_nom = reduced_drying(6.5, 1.0, 55.0)
    rows = []
    for T_c, rh_air in ((70.0, 10.0), (70.0, 20.0), (70.0, 30.0)):
        D = D0 * math.exp(-Ea / (R_GAS * (T_c + 273.15)))
        Xeq = gab_x(rh_air / 100.0, Xm, C, K)
        lo, hi = 1e-4, 2e-2                                      # calibrate the half-thickness on the nominal case
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            core, _ = slab_series(6.5, Xeq, D, mid, t_s)
            lo, hi = (mid, hi) if core < x_nom else (lo, mid)    # a thicker slab dries less
        L = 0.5 * (lo + hi)
        xr, xs, xm_, awr, aws = [], [], [], [], []
        dmv, x0v = [], []
        for tr in trials[:250]:
            X0, dm = tr["drying_X0_db"], tr["drying_Deff_mult"]
            xr.append(reduced_drying(X0, dm, tr["drying_RH_amb_pct"]))
            core, mean = slab_series(X0, Xeq, D * dm, L, t_s)
            xs.append(core); xm_.append(mean)
            awr.append(gab_aw(xr[-1], Xm, C, K)); aws.append(gab_aw(core, Xm, C, K))
            dmv.append(math.log(dm)); x0v.append(math.log(X0))
        xr, xs, xm_ = np.array(xr), np.array(xs), np.array(xm_)
        awr, aws = np.array(awr), np.array(aws)
        rank = lambda a: np.argsort(np.argsort(a)).astype(float)
        slope = lambda y: float(np.polyfit(dmv, np.log(y), 1)[0])
        rows.append({"air_T_C": T_c, "air_RH_pct": rh_air, "calibrated_half_thickness_mm": round(L * 1000, 3),
                     "Xf_core_reduced_mean": round(xr.mean(), 4), "Xf_core_reduced_sd": round(xr.std(), 4),
                     "Xf_core_resolved_mean": round(xs.mean(), 4), "Xf_core_resolved_sd": round(xs.std(), 4),
                     "Xf_mean_slab_resolved_mean": round(xm_.mean(), 4),
                     "RMSE_core_kg_kg": round(float(np.sqrt(np.mean((xr - xs) ** 2))), 4),
                     "spearman_rank_correlation": round(float(np.corrcoef(rank(xr), rank(xs))[0, 1]), 3),
                     "dlnXf_dlnDmult_reduced": round(slope(xr), 3), "dlnXf_dlnDmult_resolved": round(slope(xs), 3),
                     "aw_violations_reduced_pct": round(float(np.mean(awr > aw_max) * 100), 1),
                     "aw_violations_resolved_pct": round(float(np.mean(aws > aw_max) * 100), 1),
                     "evidence_status": "model_to_model_reduction_check_constant_diffusivity_no_shrinkage_assumed_air_state"})
    return rows


# =============================================================================== Fermentation (Chapter 12, Scenario IIIA)
FERM = [("1_Open_Loop_Exponential_Feed_PID", 1.15, 0.00, True, False),
        ("2_Assigned_Recurrent_Design_Slot_Not_Trained", 1.09, 0.40, False, False),
        ("3_Fixed_Mechanistic_ODE_Nominal_EKF_NMPC", 1.03, 0.72, True, False),
        ("4_Trustworthy_IVVO_DT_Algorithm_12_1", 0.985, 0.96, True, True)]


def ferm_trial(tr, P, feed_aggr, crab, conf, dt):
    qO2_nom = P[("Scenario_IIIA_Fermentation", "qO2_max_ferm")]
    YXS_ox, YXS_red = P[("Scenario_IIIA_Fermentation", "YXS_ox_ferm")], P[("Scenario_IIIA_Fermentation", "YXS_red_ferm")]
    YO2, K_iP, d_dil = 0.397, 12.5, 0.0195   # coded constants of run_all_benchmarks.py (YO2 assigned; stoichiometric value about 0.408)
    cx0 = tr["ferm_CX0_g_L"]; qo2_m = qO2_nom * tr["ferm_qO2_max_mult"]; kla_m = tr["ferm_kLa_mult"]
    cx, etoh, s_fed, do_viol, cool_viol = cx0, 0.0, 0.0, 0, 0
    for step in range(int(round(28.0 / dt))):
        t = step * dt
        xi = kla_m * (0.76 + 0.22 * crab if t >= 12.0 else 1.0)
        otr = (3.15 + 1.05 * crab) * xi
        qO2_cell = qo2_m * (K_iP / (K_iP + etoh))
        qO2_av = min(qO2_cell, otr / max(cx, 1.0))
        qS_ox_max = qO2_av / YO2
        qS_sched = min(qO2_nom, 3.55 / max(cx, 1.0)) / YO2
        qS_crit = qS_sched * (1.0 - crab) + qS_ox_max * crab
        qS_app = qS_crit * feed_aggr
        qS_ox = min(qS_app, qS_ox_max * (0.981 if conf else 1.0))
        qS_red = max(0.0, qS_app - qS_ox)
        our = qS_ox * YO2 * cx
        if our > otr * 0.96 and not conf:
            do_viol = 1
        q_metab = our * 0.46 + qS_red * cx * 0.15
        q_lim = (1.62 if 18.0 <= t < 24.0 else 1.98) * (1.45 if conf else (1.0 + 0.08 * crab))
        k_d = 0.003
        if q_metab > q_lim:
            cool_viol = 1
            k_d = 0.016 * (1.0 - 0.65 * crab)
        mu = YXS_ox * qS_ox + YXS_red * qS_red - k_d
        cx_next = cx * math.exp((mu - d_dil) * dt)
        s_fed += qS_app * 0.5 * (cx + cx_next) * dt
        etoh = max(0.0, etoh * math.exp(-d_dil * dt) + 0.48 * qS_red * 0.5 * (cx + cx_next) * dt)
        cx = cx_next
    yxs = (cx * 1.72 - cx0) / max(1.0, s_fed * 1.38)
    kwh = 2.84 * ((71.4 / max(cx, 35.0)) ** 0.85) * (1.0 - 0.14 * crab)
    return cx, yxs, etoh, do_viol, cool_viol, kwh


def ferm_check(P, trials):
    stored = {}
    with open(RESULTS / "Ch12_Table_12_3_Scenario_IIIA_Fermentation_Benchmark.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            stored[r["control_strategy"]] = r
    rows = []
    for name, feed, crab, _stoich, conf in FERM:
        base = None
        for dt in (1.0, 0.25, 0.05):
            res = np.array([ferm_trial(tr, P, feed, crab, conf, dt) for tr in trials[:250]])
            m = res.mean(axis=0)
            if base is None:
                base = m
            s = stored[name]
            rows.append({"strategy": name, "time_step_h": dt,
                         "final_biomass_g_L": round(m[0], 2), "yield_YXS_g_g": round(m[1], 3), "ethanol_g_L": round(m[2], 2),
                         "DO_starvation_pct": round(m[3] * 100, 1), "cooling_saturation_pct": round(m[4] * 100, 1),
                         "energy_kWh_kgX": round(m[5], 2),
                         "biomass_change_vs_1h_pct": round((m[0] / base[0] - 1) * 100, 2),
                         "ethanol_change_vs_1h_g_L": round(m[2] - base[2], 2),
                         "matches_stored_csv": (round(m[0], 1) == float(s["final_biomass_C_X_g_per_L_mean"])) if dt == 1.0 else "",
                         "evidence_status": "model_to_model_time_step_reduction_check"})
    return rows


def main():
    P, trials = load_params(), load_trials()
    for name, fn, arg in (("Ch10_UHT_Reduction_Check.csv", uht_check, trials),
                          ("Ch11_Drying_Reduction_Check.csv", lambda t: drying_check(P, t), trials),
                          ("Ch12_Fermentation_Reduction_Check.csv", lambda t: ferm_check(P, t), trials)):
        rows = fn(arg)
        write(name, rows)
        print(name)
        for r in rows:
            print("  ", {k: v for k, v in r.items() if k != "evidence_status"})


if __name__ == "__main__":
    main()
