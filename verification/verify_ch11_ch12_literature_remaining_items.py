"""Literature comparison for the quantities of Chapters 11 and 12 that had no measured counterpart in the first checks.

Sources (data/external; rows read from tables are exact, rows read from figures carry the stated reading uncertainty):
  Extruder dynamics   Kumar et al., J. Food Eng. 75 (2006) 237-244 (library 310): residence-time distributions of a co-rotating twin-screw laboratory extruder
                      (Table 2 exact; first tracer appearance read from Fig. 7, about +-3 s).
  Expansion index     Della Valle et al., J. Food Eng. 31 (1997) 271-296 (library 157), Fig. 8b: sectional expansion index of starch B at three moisture contents
                      (read from the figure, about +-0.1).
  Tunnel oven         Mayo Bayon et al., Sensors 14 (2014) 14712-14731: air temperature along a 40 m electric biscuit tunnel oven with four sections, four loggers
                      across the belt (Fig. 18, read to about +-10 degC); oven parameters from Table 2.
  Heat-pump dryer     Tunckal, Coskun & Erdogan, J. Therm. Sci. Technol. 36 (2016) 161-172: closed-loop heat-pump dryer (750 W compressor), apple slices 5 mm at 40 degC
                      (Tables 4-6, exact).
  Cold chain          Koutsoumanis et al., Appl. Environ. Microbiol. 66 (2000) 3528-3534: growth rate of pseudomonads on red mullet at 0-20 degC (Table 2, exact);
                      Mercier et al., Compr. Rev. Food Sci. Food Saf. 16 (2017) 647-667 (library 358): reported temperature differences within the chain (text).

  Extruder control    extruder_delay_control(): executed delay-based simulation of a +2 pp feed-moisture step through the measured
                      residence-time density (dead time + gamma), comparing no correction, PI feedback on a die moisture measurement,
                      ideal die feedback and feed-forward from a feeder moisture measurement (exact and +-10 % gain) with PI trim.
                      Sensor lags (2 s), the water-injection position and the gain errors are assumptions of this check.

Scope: parameter plausibility and physical bounds only. None of these studies is a validation of a scenario. The scrap rate has no measured counterpart.
Output: Ch11_Ch12_Literature_Remaining_Check.csv.   Run: python verify_ch11_ch12_literature_remaining_items.py
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import gamma

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v):
    return float(v) if v not in ("", None) else float("nan")


def extruder_settling(rows):
    d = read_csv(DATA / "external" / "extruder_residence_time_Kumar2006.csv")
    mrt = [num(r["mean_residence_time_s"]) for r in d]
    t95 = [gamma(num(r["number_of_CSTRs"]), scale=num(r["mean_residence_time_s"]) / num(r["number_of_CSTRs"])).ppf(0.95) for r in d]
    dead = [num(r["first_tracer_appearance_s_read_from_figure"]) for r in d if r["first_tracer_appearance_s_read_from_figure"]]
    rows.append({"item": "Extruder_residence_time_measured", "mean_residence_time_s_min_max": f"{min(mrt):.1f}-{max(mrt):.1f}",
                 "first_tracer_appearance_s_min_max": f"{min(dead):.0f}-{max(dead):.0f}", "time_to_95pct_of_a_feed_step_s_tank_in_series_min_max": f"{min(t95):.0f}-{max(t95):.0f}",
                 "pilot_open_loop_steady_state_s_DellaValle1987": 600})
    for slot, t in (("1_fixed_recipe", 420.0), ("2_open_loop", 77.8), ("3_black_box", 22.2), ("4_twin", 7.7)):
        rows.append({"item": "Extruder_settling_scenario_vs_dead_time_" + slot, "scenario_settling_s": t, "below_shortest_first_appearance": t < min(dead),
                     "below_mean_residence_time_range": t < min(mrt), "within_first_appearance_range_or_above_95pct_time": t >= min(dead)})


def _plant(theta, mean, var, dt, n):
    """Residence-time density of the Kumar extruder as dead time + gamma (same mean and variance as Table 2, first appearance theta)."""
    m = mean - theta
    k, scale = m * m / var, var / m
    t = np.arange(n) * dt
    E = np.where(t > theta, gamma(k, scale=scale).pdf(np.maximum(t - theta, 0.0)), 0.0)
    return E / (E.sum() * dt), k, scale


def _closed_loop(E, dt, n, step, mode, Kc=0.0, tauI=1.0, g_ff=1.0, tau_s=2.0, u_min=-4.5):
    """Feed-moisture step `step` (percentage points) at the feeder at t = 0. Water injection u (pp) enters with the solids, so it follows
    the same residence-time density E. Die melt moisture y = (E * (d + u))(t). Sensors: die moisture and feeder moisture, first-order lag tau_s.
    mode: none | pi (feedback on die moisture) | ideal_fb (exact cancellation from the instant the die sensor detects 1 % of the step)
          | ff (feed-forward from the feeder sensor with gain g_ff, plus PI trim on the die sensor when Kc > 0)."""
    v = np.zeros(n); y = np.zeros(n)
    ym = dm = integ = 0.0; detected = False
    a = dt / tau_s
    for k in range(n):
        if k:
            y[k] = dt * float(np.dot(E[1:k + 1], v[k - 1::-1]))
        ym += a * (y[k] - ym); dm += a * (step - dm)
        u = 0.0
        if mode in ("pi", "ff") and Kc > 0.0:
            e = -ym; integ += e * dt
            u += Kc * (e + integ / tauI)
        if mode == "ff":
            u += -g_ff * dm
        if mode == "ideal_fb":
            detected = detected or ym > 0.01 * step
            u = -step if detected else 0.0
        u = max(u_min, u)
        v[k] = step + u
    return y


def _metrics(y, dt, band_settle, bands):
    out = {}
    above = np.nonzero(np.abs(y) > band_settle)[0]
    if above.size == 0:
        out["settling_time_s"] = 0.0
    elif above[-1] == len(y) - 1:
        out["settling_time_s"] = None
    else:
        out["settling_time_s"] = round((above[-1] + 1) * dt, 1)
    out["peak_die_moisture_deviation_pp"] = round(float(np.abs(y).max()), 3)
    for name, b in bands.items():
        out[f"time_outside_{name}_s"] = round(float((np.abs(y) > b).sum() * dt), 1)
    return out


def extruder_delay_control(rows):
    """Executed delay-based disturbance simulation for Scenario IIB (feedback on the die versus feed-forward from the feeder).

    Disturbance: +2.0 percentage points of feed moisture at the feeder (the largest step declared in the scenario, +0.020 w.b.).
    Sensor positions: (1) moisture analyser at the solids-feeder discharge (upstream information, lag 2 s, assumed); (2) water-injection
    port in the first barrel section, so that injected water follows the residence-time density of the solids; (3) melt-moisture
    measurement at the die (lag 2 s, assumed), which receives the disturbance only after the transport delay of the extruder.
    Plant: dead time (first tracer appearance) plus gamma distribution with the measured mean and variance (Kumar et al. 2006).
    Settled variable: die melt moisture; settled when it stays within 5 % of the step (0.1 pp) to the end of the 600 s horizon, time from
    the step at the feeder. Product tolerance: the assigned expansion band 4.2 +- 0.145 translated to die moisture at the nominal
    0.165 w.b. with the measured (-1.75) and the assigned (-0.86) logarithmic moisture sensitivity of the expansion index.
    PI gains: grid search for the shortest settling time (best case for PI feedback). Gain errors of the feed-forward: +-10 % (assumed)."""
    d = [r for r in read_csv(DATA / "external" / "extruder_residence_time_Kumar2006.csv") if r["first_tracer_appearance_s_read_from_figure"]]
    dt, horizon, step, w_nom = 0.5, 600.0, 2.0, 0.165
    n = int(horizon / dt)
    bands = {"band_measured_SEI_sensitivity": 100 * w_nom * 0.145 / (4.2 * 1.75), "band_assigned_SEI_sensitivity": 100 * w_nom * 0.145 / (4.2 * 0.86)}
    band_settle = 0.05 * step
    for r in d:
        theta, mean, var = num(r["first_tracer_appearance_s_read_from_figure"]), num(r["mean_residence_time_s"]), num(r["variance_s2"])
        E, k, scale = _plant(theta, mean, var, dt, n)
        base = {"item": f"Extruder_delay_simulation_{int(num(r['screw_speed_rpm']))}rpm", "dead_time_s": theta, "mean_residence_time_s": mean,
                "gamma_shape_after_dead_time": round(k, 2), "step_pp": step, "settling_band_pp": band_settle,
                **{f"{kk}_pp": round(vv, 2) for kk, vv in bands.items()}}
        res = {"none": _metrics(_closed_loop(E, dt, n, step, "none"), dt, band_settle, bands)}
        best = None
        for Kc in np.linspace(0.05, 1.5, 30):
            for tauI in np.linspace(5.0, 200.0, 40):
                st = _metrics(_closed_loop(E, dt, n, step, "pi", Kc, tauI), dt, band_settle, bands)["settling_time_s"]
                if st is not None and (best is None or st < best[0]):
                    best = (st, Kc, tauI)
        _, Kc, tauI = best
        res["pi_feedback_die"] = _metrics(_closed_loop(E, dt, n, step, "pi", Kc, tauI), dt, band_settle, bands)
        res["ideal_feedback_die"] = _metrics(_closed_loop(E, dt, n, step, "ideal_fb"), dt, band_settle, bands)
        res["feedforward_exact"] = _metrics(_closed_loop(E, dt, n, step, "ff", 0.0, 1.0, 1.0), dt, band_settle, bands)
        for g in (0.9, 1.1):
            res[f"feedforward_gain_{g}_with_pi_trim"] = _metrics(_closed_loop(E, dt, n, step, "ff", Kc, tauI, g), dt, band_settle, bands)
        for name, m in res.items():
            rows.append({**base, "architecture": name, "pi_Kc": round(Kc, 3) if "pi" in name else "", "pi_tauI_s": round(tauI, 1) if "pi" in name else "", **m})
            print(rows[-1], flush=True)


def sei(rows):
    d = read_csv(DATA / "external" / "extrusion_sei_moisture_DellaValle1997.csv")
    mc = np.array([num(r["moisture_content_wet_basis"]) for r in d]); s = np.array([num(r["sectional_expansion_index_read_from_figure"]) for r in d])
    levels = sorted(set(mc)); means = [float(s[mc == m].mean()) for m in levels]
    slope = float(np.polyfit(np.log(mc), np.log(s), 1)[0])
    rows.append({"item": "Expansion_index_moisture_sensitivity_measured", "moisture_levels": "/".join(f"{m:.3f}" for m in levels), "mean_SEI_per_level": "/".join(f"{m:.2f}" for m in means),
                 "log_log_slope_SEI_vs_moisture": round(slope, 2), "points": len(s)})
    tr = read_csv(DATA / "monte_carlo_disturbances_seed42.csv")[:250]
    x = np.array([num(t["extrusion_X_feed_wb"]) for t in tr]); k0 = np.array([num(t["extrusion_K0_mult"]) for t in tr])
    sme = 548.0 * k0 * (0.18 / x) ** 0.82
    scen = 3.98 * (sme / 520.0) ** 0.34 * (0.18 / x) ** 0.58
    net_scen = -(0.58 + 0.34 * 0.82)
    b = -slope - 0.34 * 0.82
    alt = 3.98 * (sme / 520.0) ** 0.34 * (0.18 / x) ** b
    for name, v in (("assigned", scen), ("measured_moisture_sensitivity", alt)):
        rows.append({"item": "Expansion_index_fixed_recipe_" + name, "mean_SEI": round(float(v.mean()), 2), "sd_SEI": round(float(v.std(ddof=1)), 2),
                     "scrap_pct_band_4.2_pm_0.145": round(float((abs(v - 4.2) > 0.145).mean() * 100), 1), "net_moisture_exponent": round(net_scen if name == "assigned" else slope, 2)})


def oven(rows):
    d = read_csv(DATA / "external" / "tunnel_oven_electric_air_temperature_Mayo2014.csv")
    pos = [num(r["position_m"]) for r in d]
    spread = {p: max(num(r[f"row_{i}_C"]) for i in range(1, 5)) - min(num(r[f"row_{i}_C"]) for i in range(1, 5)) for p, r in zip(pos, d)}
    mean = {p: float(np.mean([num(r[f"row_{i}_C"]) for i in range(1, 5)])) for p, r in zip(pos, d)}
    end = {10: 180, 20: 230, 30: 240, 40: 190}
    rows.append({"item": "Tunnel_oven_measured_profile", "oven_length_m": 40, "sections": 4, "residence_min": 3.15, "belt_speed_m_per_min": round(40 / 3.15, 1),
                 "max_row_spread_C": max(spread.values()), "row_spread_C_at_20_25_30_35_m": "/".join(f"{spread[p]:.0f}" for p in (20, 25, 30, 35)),
                 "mean_temperature_C_at_section_ends_10_20_30_40_m": "/".join(f"{mean[p]:.0f}" for p in end), "setpoints_C": "/".join(str(v) for v in end.values()),
                 "shortfall_C_at_section_ends": "/".join(f"{end[p] - mean[p]:.0f}" for p in end), "peak_measured_C": max(max(num(r[f'row_{i}_C']) for i in range(1, 5)) for r in d)})
    rows.append({"item": "Tunnel_oven_scenario_vs_measured", "scenario_length_m": 45, "scenario_residence_min": 22.0, "scenario_belt_speed_m_per_min": round(45 / 22.0, 2),
                 "belt_speed_ratio_measured_to_scenario": round((40 / 3.15) / (45 / 22.0), 1), "published_biscuit_baking_time_min_Hadiyanto": 13.3, "scenario_zone_setpoints_declared": "no"})


def dryer(rows):
    d = read_csv(DATA / "external" / "heat_pump_dryer_apple_Tunckal2016.csv")
    e = np.array([num(r["total_energy_kWh"]) for r in d]); s = np.array([num(r["SMER_kg_per_kWh"]) for r in d]); cop = np.array([num(r["COP_system_without_external_condenser"]) for r in d])
    water = float((s * e).sum())
    rows.append({"item": "Heat_pump_dryer_apple_measured", "energy_kWh": round(float(e.sum()), 3), "water_removed_kg": round(water, 2), "SMER_mean_kg_per_kWh": round(water / float(e.sum()), 3),
                 "SMER_first_hour": s[0], "SMER_last_hour": s[-1], "SMER_decline_factor": round(float(s[0] / s[-1]), 1), "COP_system_range": f"{cop.min():.2f}-{cop.max():.2f}",
                 "scale": "laboratory 750 W compressor, 40 degC, 5 mm slices"})
    for slot, g, tg in (("1_fixed_recipe", 0.0, False), ("2_open_loop", 0.38, False), ("3_black_box", 0.72, False), ("4_twin", 0.92, True)):
        cope = 1.64 + 1.48 * g + (0.58 if tg else 0.0)
        smer = {"1_fixed_recipe": 0.67, "2_open_loop": 0.89, "3_black_box": 1.10, "4_twin": 1.45}[slot]
        rows.append({"item": "Heat_pump_dryer_scenario_vs_measured_" + slot, "scenario_SMER_kg_per_kWh": smer, "ratio_to_measured_mean": round(smer / (water / float(e.sum())), 1), "scenario_COP_eff": round(cope, 2),
                     "measured_COP_range": f"{cop.min():.2f}-{cop.max():.2f}"})


def coldchain(rows):
    d = read_csv(DATA / "external" / "pseudomonas_growth_rate_Koutsoumanis2000.csv")
    T = np.array([num(r["temperature_C"]) for r in d]) + 273.15; mu = np.array([num(r["mu_max_per_h_aerobic_red_mullet_pseudomonads"]) for r in d])
    slope, icpt = np.polyfit(1.0 / T, np.log(mu), 1)
    R = 8.314462618
    ea = -slope * R
    mu4 = math.exp(icpt + slope / 277.15)
    mu_s = 0.048
    ea_s = 68500.0
    rsl_meas = (7.0 - 3.2) * math.log(10.0) / mu4
    rsl_scen = (7.0 - 3.2) * math.log(10.0) / mu_s
    rows.append({"item": "Pseudomonas_growth_measured_vs_scenario", "Ea_fit_kJ_mol": round(ea / 1000, 1), "Ea_scenario_kJ_mol": 68.5, "Ea_reported_table_kJ_mol": "65.4 +- 5.5 (abstract: 75-85 for the four groups)",
                 "mu_4C_fit_per_h": round(mu4, 3), "mu_4C_scenario_per_h": mu_s, "ratio_scenario_to_fit": round(mu_s / mu4, 2),
                 "time_3.2_to_7_log_at_4C_h_fit": round(rsl_meas, 0), "time_3.2_to_7_log_at_4C_h_scenario": round(rsl_scen, 0),
                 "growth_ratio_8C_over_4C_fit": round(math.exp(-slope * (1 / 277.15 - 1 / 281.15)), 2), "growth_ratio_8C_over_4C_scenario": round(math.exp(-(-ea_s / R) * (1 / 277.15 - 1 / 281.15)) , 2)})
    co2 = read_csv(DATA / "external" / "pseudomonas_growth_rate_CO2_4C_Koutsoumanis2000.csv")
    tr = read_csv(DATA / "monte_carlo_disturbances_seed42.csv")[:240]
    pol = (("1_static_FIFO", 0.00, False, False), ("2_IoT_surface", 0.48, False, False), ("3_zonal_ROM", 0.88, False, False), ("4_digital_twin", 0.945, True, True))
    for name, rom, hsi, conf in pol:
        for label, mu_ref in (("assigned_mu", mu_s),) + tuple((f"measured_{int(num(r['CO2_percent']))}pct_CO2", num(r["mu_max_per_h_pseudomonads_red_mullet"])) for r in co2 if num(r["CO2_percent"]) <= 50):
            spoil = 0
            for i, t in enumerate(tr):
                dT = 3.8 + max(-0.8, num(t["coldchain_T_amb_offset_C"]) * 0.95)
                apron = 1.65 if i % 8 == 0 else 0.0
                Tt = 2.1 + 0.55 * dT + apron
                Te = 2.1 + rom * (0.55 * dT) + (0.95 if hsi else 0.25 * rom) * apron
                n0 = num(t["coldchain_N0_SSO_log10"])
                growth = lambda T_: (mu_ref / math.log(10.0)) * math.exp(-(ea_s / R) * (1.0 / (T_ + 273.15) - 1.0 / 277.15))
                n0e = (n0 + 0.04 * (n0 - 3.20)) if hsi else (3.20 + 0.22 * (n0 - 3.20))
                rsl_true = (7.0 - n0) / growth(Tt)
                rsl_est = (7.0 - n0e) / growth(Te)
                q = 14.5 if conf else (18.5 if rom == 0.48 else (9.5 if rom > 0.8 else 0.5))
                eff = 0.81 if conf else (0.72 if rom > 0.8 else (0.45 if rom > 0.3 else 0.0))
                t_req = 146.0 - eff * max(-30.0, min(62.0, (162.0 - rsl_true) * 0.92))
                if n0 + growth(Tt) * t_req > 7.0:
                    spoil += 1
            rows.append({"item": f"Cold_chain_spoilage_{label}_{name}", "mu_4C_per_h": round(mu_ref, 3), "spoilage_share_pct": round(spoil / len(tr) * 100, 2)})
    rows.append({"item": "Cold_chain_spatial_temperature_literature", "reported_difference_between_pallets_before_loading_C": 2.0, "reported_difference_between_two_meat_packages_in_a_display_cabinet_C": 4.0,
                 "scenario_core_to_surface_range_C": 3.8, "scenario_spatial_range_inside_reported_order": True, "pallet_level_product_quality_data": "none in the collection"})


def main():
    rows = []
    extruder_settling(rows); extruder_delay_control(rows); sei(rows); oven(rows); dryer(rows); coldchain(rows)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    keys = ["item"] + [k for k in keys if k != "item"]
    with open(HERE / "Ch11_Ch12_Literature_Remaining_Check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["evidence_status"]); w.writeheader()
        for r in rows:
            r["evidence_status"] = ("executed_delay_simulation_on_measured_residence_time_density_assumed_sensor_positions_and_lags_not_validation"
                                    if r["item"].startswith("Extruder_delay_simulation") else
                                    "published_laboratory_or_industrial_data_parameter_plausibility_not_validation")
            w.writerow(r); print(r)


if __name__ == "__main__":
    main()
