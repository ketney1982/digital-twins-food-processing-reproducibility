"""Comparison of scenario outputs with industrial-scale and pilot-scale data from the reference collection (Chapters 10 and 11).

Sources (all in the monograph's reference full-text collection and in the EndNote library):
  UHT         Petermeier et al., J. Food Eng. 55 (2002) 9-17 (record 409): measured overall heat-transfer coefficient of an industrial
              tubular UHT plant (about 15 000 kg/h) over a 10 h run, Fig. 3; the two end points were extracted from the vector figure
              (data/external/uht_plant_heat_transfer_coefficient_Petermeier2002.csv).
  Bread oven  Zisopoulos et al., J. Food Eng. (2015) (record in library; file 528): operating conditions of an industrial continuous
              bread oven (235 degC for 35 min, white bread 0.8 kg; par-baked 160 degC for 30 min) and the stated energy distribution (25 % to
              the product); an oven energy estimate follows from its assumptions. These are modelling assumptions of a chain study, not
              measurements of a plant.
  Acrylamide  Goekmen & Senyuva, Food Chem. 99 (2006) 238-243 (record in library; file 206): acrylamide in wheat flour heated at 150, 180 and
              220 degC (Fig. 2; values read from an enlarged rendering, about +-10 ng/g; data/external/acrylamide_wheat_flour_Gokmen2006.csv);
              De Vleeschouwer et al., Food Chem. 111 (2008) 719-729 (file 130): activation energies of acrylamide formation in
              asparagine-glucose systems.
  Biscuit     Hadiyanto et al., Innov. Food Sci. Emerg. Technol. 8 (2007) 285-298 (record 238): biscuit properties and baking time
              (Table 5: half-thickness 2.5 mm, 200 degC, 800 s, k = 0.2 W/(m K), density 540 kg/m3, cp = 1712.5 J/(kg K), h = 15 W/(m2 K), water 0.19).
  Extrusion   Della Valle, Tayeb & Melcion, J. Food Eng. 6 (1987) 423-444 (record 156): pilot-scale twin-screw extruder (Clextral BC 45):
              die pressure 30-70 bar, steady state after about 10 min, pressure fluctuation about 6 bar; Della Valle et al., J. Food Eng.
              31 (1997) (record 157), Tables 1-2: specific mechanical energy 130-374 Wh/kg.

Scope: one industrial UHT plant, an assumed industrial bread chain, a flour heating experiment in vials, a published biscuit simulation set-up and a
pilot extruder. No complete industrial data set for the scenarios exists in the collection.
Output: Ch10_Ch11_Industrial_Pilot_Check.csv.   Run: python verify_ch10_ch11_industrial_pilot_data.py
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def read_csv(name):
    with open(DATA / "external" / name, newline="", encoding="utf-8") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def trials():
    with open(DATA / "monte_carlo_disturbances_seed42.csv", newline="", encoding="utf-8") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def uht(rows):
    u = read_csv("uht_plant_heat_transfer_coefficient_Petermeier2002.csv")
    u0, u1, t1 = u[0]["overall_heat_transfer_coefficient_W_m2K"], u[1]["overall_heat_transfer_coefficient_W_m2K"], u[1]["time_h"]
    rf = 1.0 / u1 - 1.0 / u0
    plant_rate = rf / t1
    tr = trials()[:200]
    mult = float(np.mean([t["uht_k_foul_mult"] * t["uht_protein_load_mult"] ** 1.25 for t in tr]))
    rows.append({"item": "UHT_industrial_plant_measured", "U_start_W_m2K": u0, "U_end_W_m2K": u1, "run_h": t1, "U_decline_pct": round((1 - u1 / u0) * 100, 1),
                 "fouling_resistance_end_m2K_W": round(rf, 7), "fouling_rate_m2K_W_per_h": round(plant_rate, 8),
                 "hours_to_5e-4_at_same_linear_rate": round(5e-4 / plant_rate, 1)})
    for name, beta in (("1_fixed_setpoint", 0.25), ("2_open_loop_prehold", 0.68), ("3_fouling_aware_twin", 0.76)):
        k = 0.000081 * (1.0 - 0.68 * beta) * mult
        rows.append({"item": "UHT_scenario_assigned_rate_" + name, "fouling_rate_m2K_W_per_h": round(k, 8), "ratio_to_plant_rate": round(k / plant_rate, 1),
                     "hours_to_5e-4": round(5e-4 / k, 1)})


def baking(rows):
    w_final, evap = 0.376, 0.10
    w0 = w_final / (1.0 - evap * (1.0 - w_final))                      # dough water fraction from 10 % evaporated and 37.6 % final moisture
    cp = 3.0                                                            # kJ/(kg K), assumed
    sens = (0.9 * (100 - 28) + 0.1 * (180 - 28)) * cp                   # crumb to 100 degC, crust (10 %) to 180 degC
    latent = w0 * evap * 2260.0
    oven_energy = (sens + latent) / 0.25 / 1000.0                       # 25 % of the oven energy goes to the product
    rows.append({"item": "Bread_industrial_oven_conditions_from_chain_study", "white_bread_oven_C": 235, "white_bread_time_min": 35, "par_baked_oven_C": 160,
                 "par_baked_time_min": 30, "scenario_residence_time_text_min": 22.0, "scenario_residence_time_code_min": 18.2})
    rows.append({"item": "Bread_oven_energy_estimate_from_the_study_assumptions", "dough_water_fraction": round(w0, 3), "product_side_MJ_per_kg": round((sens + latent) / 1000.0, 3),
                 "oven_energy_MJ_per_kg": round(oven_energy, 2), "scenario_SEC_range_MJ_per_kg": "1.80-2.18",
                 "ratio_scenario_to_estimate_low_high": f"{1.80 / oven_energy:.2f}-{2.18 / oven_energy:.2f}"})


def acrylamide(rows):
    d = read_csv("acrylamide_wheat_flour_Gokmen2006.csv")
    peak = {T: max(r[f"acrylamide_{T}C_ng_g"] for r in d) for T in (150, 180, 220)}
    at15 = {T: [r for r in d if r["time_min"] == 15][0][f"acrylamide_{T}C_ng_g"] for T in (180, 220)}
    dT = 220 - 180
    sens = math.log(peak[220] / peak[180]) / dT
    R = 8.314462618
    rate_ratio = math.exp(118000.0 / R * (1 / 453.15 - 1 / 493.15))
    rows.append({"item": "Acrylamide_wheat_flour_measured", "peak_150C_ng_g": peak[150], "peak_180C_ng_g": peak[180], "peak_220C_ng_g": peak[220],
                 "peak_ratio_220_to_180": round(peak[220] / peak[180], 2), "dlnAA_dT_per_K_between_180_and_220C": round(sens, 4),
                 "ratio_at_15min_220_to_180": round(at15[220] / at15[180], 1), "Arrhenius_rate_ratio_118kJ_mol_220_to_180": round(rate_ratio, 1)})
    rows.append({"item": "Acrylamide_scenario_vs_measured", "scenario_fixed_recipe_mean_ug_kg": 274.2, "scenario_fixed_recipe_sd_ug_kg": 67.7,
                 "measured_peak_range_180_to_220C_ng_g": f"{peak[180]:.0f}-{peak[220]:.0f}"})
    rows.append({"item": "Acrylamide_exponent_on_heat_transfer_multiplier", "assigned_exponent": 1.35, "implied_surface_dT_per_unit_ln_h_K_from_measured_net_sensitivity": round(1.35 / sens, 0),
                 "exponent_if_surface_T_changes_20K_per_unit_ln_h": round(sens * 20.0, 2), "exponent_if_surface_T_changes_40K_per_unit_ln_h": round(sens * 40.0, 2),
                 "declared_Ea_kJ_mol": 118, "measured_formation_Ea_kJ_mol_asparagine_glucose": "147-181"})


def biscuit(rows):
    rho, cp, k, h, w0, Tov = 540.0, 1712.5, 0.2, 15.0, 0.19, 200.0
    lv = 2.26e6
    for half in (0.0025, 0.0042):
        n, dt = 20, 1.0
        dx = half / n
        T = np.full(n, 20.0)
        mass = rho * dx                                          # kg/m2 per cell
        latent = lv * w0 * mass                                  # J/m2 per cell over 99-101 degC (all water evaporable)
        t, t_dry, t_180 = 0.0, None, None
        while t < 3000.0:
            c = mass * cp + np.where((T > 99.0) & (T < 101.0), latent / 2.0, 0.0)
            a = k / dx
            lower = np.full(n, -a); diag = np.full(n, 2 * a); upper = np.full(n, -a)
            g = 1.0 / (1.0 / h + dx / (2 * k))
            diag[0] = a; diag[-1] = a + g
            rhs = c / dt * T; rhs[-1] += g * Tov
            ab = np.vstack([np.r_[0.0, upper[:-1]], diag + c / dt, np.r_[lower[1:], 0.0]])
            T = solve_banded((1, 1), ab, rhs); t += dt
            if t_dry is None and T[0] > 101.0:
                t_dry = t
            if t_180 is None and T[0] >= 180.0:
                t_180 = t
            if t_dry and t_180:
                break
        rows.append({"item": f"Biscuit_Hadiyanto_properties_half_thickness_{half * 1000:.1f}mm", "centre_leaves_100C_plateau_s": t_dry, "centre_reaches_180C_s": t_180,
                     "published_baking_time_s_for_2.5mm": 800, "scenario_residence_time_min": 22.0})


def extrusion(rows):
    rows.append({"item": "Extrusion_pilot_scale_measured", "die_pressure_bar_range": "30-70", "pressure_fluctuation_bar": 6, "steady_state_time_min": 10,
                 "SME_Wh_per_kg_range": "130-374", "SME_kJ_per_kg_range": "468-1346"})
    rows.append({"item": "Extrusion_scenario_vs_pilot", "scenario_die_pressure_MPa_fixed_recipe": 6.55, "inside_measured_range": True, "scenario_interlock_MPa": 12.0,
                 "interlock_above_measured_maximum_MPa": 7.0, "scenario_SME_kJ_kg_fixed_recipe": 557.3, "SME_inside_measured_range": True,
                 "scenario_settling_time_fixed_recipe_s": 420.0, "measured_open_loop_steady_state_s": 600.0, "ratio_scenario_to_measured": 0.7,
                 "scenario_settling_time_controlled_slots_s": "7.7-77.8 (no measured counterpart)"})


def main():
    rows = []
    uht(rows); baking(rows); acrylamide(rows); biscuit(rows); extrusion(rows)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "item", k))
    with open(HERE / "Ch10_Ch11_Industrial_Pilot_Check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["evidence_status"]); w.writeheader()
        for r in rows:
            r["evidence_status"] = "published_industrial_or_pilot_data_one_plant_each_not_complete_industrial_validation"
            w.writerow(r); print(r)


if __name__ == "__main__":
    main()
