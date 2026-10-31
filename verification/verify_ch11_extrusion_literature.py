"""Literature check of the extrusion scenario (Chapter 11, Scenario IIB) against measured starch rheology.

Data: Table 1 of Vergnes & Villemaire, Rheol. Acta 26 (1987) 570-576 (library record 549): power-law parameters (index m, consistency K) of
molten maize starch after a standard pre-shear (400 rpm, 40 s) at 110, 150 and 170 degC and 21, 28 and 33 % water (wet basis); transcribed in
data/external/molten_maize_starch_rheology_Vergnes1987.csv. The paper describes K by K = K0' exp[E/R (1/T - 1/T0)] exp[-a (MC - MC0)].

Checks: (i) the declared consistency K0 = 14200 Pa s^n and index n = 0.34 (parameter table) against the measured range at 110 degC and 28 %;
(ii) the moisture sensitivity of the die pressure and the specific mechanical energy in the reduced model (exponents 1.10 and 0.82 on 0.18/x for the
fixed recipe) against the measured exponential moisture law, over the 250 Monte Carlo trials. At fixed throughput, geometry and index the die pressure drop
and the viscous-dissipation power are proportional to the consistency K.

Scope: one starch, a laboratory rheometer; the pressure and energy comparison assumes the scenario material follows this law.
Output: Ch11_Extrusion_Literature_Check.csv.   Run: python verify_ch11_extrusion_literature.py
"""
import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
R_GAS = 8.314462618


def main():
    rows = list(csv.DictReader(open(DATA / "external" / "molten_maize_starch_rheology_Vergnes1987.csv", newline="", encoding="utf-8")))
    T = np.array([float(r["temperature_C"]) for r in rows]) + 273.15
    mc = np.array([float(r["moisture_wet_basis_pct"]) for r in rows])
    K = np.array([float(r["consistency_K_Pa_s_m"]) for r in rows])
    m = np.array([float(r["power_law_index_m"]) for r in rows])
    A = np.c_[np.ones(len(K)), 1.0 / T - 1.0 / 383.15, -(mc - 28.0)]
    coef, *_ = np.linalg.lstsq(A, np.log(K), rcond=None)
    k_ref, e_over_r, alpha = math.exp(coef[0]), coef[1], coef[2]
    resid = np.log(K) - A @ coef
    params = {}
    with open(DATA / "kinetic_and_transport_parameters.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            params[(r["process_id"], r["symbol"])] = float(r["value"])
    k_decl, n_decl = params[("Scenario_IIB_Extrusion", "K0_extr")], params[("Scenario_IIB_Extrusion", "n_extr")]
    trials = []
    with open(DATA / "monte_carlo_disturbances_seed42.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            trials.append({k: float(v) for k, v in r.items()})
    trials = trials[:250]
    dp_red, dp_lit, s_red, s_lit = [], [], [], []
    for tr in trials:
        k0m, x = tr["extrusion_K0_mult"], tr["extrusion_X_feed_wb"]
        dp_red.append(6.40 * (k0m ** 1.25) * (0.18 / x) ** 1.10)
        dp_lit.append(6.40 * k0m * math.exp(-alpha * (100.0 * x - 18.0)))
        s_red.append(548.0 * (k0m ** 1.0) * (0.18 / x) ** 0.82)
        s_lit.append(548.0 * k0m * math.exp(-alpha * (100.0 * x - 18.0)))
    xs = np.array([t["extrusion_X_feed_wb"] for t in trials])
    out = [{"item": "fit_of_K_to_Table_1", "K_ref_110C_28pct_Pa_s_m": round(k_ref, 0), "E_over_R_K": round(e_over_r, 0),
            "moisture_coefficient_per_wt_pct": round(alpha, 4), "max_abs_log_residual": round(float(np.abs(resid).max()), 3)},
           {"item": "declared_parameters_vs_measured_110C_28pct", "declared_K0_Pa_s_n": k_decl, "declared_n": n_decl,
            "measured_K_Pa_s_m": 18400, "measured_m": 0.33, "ratio_declared_to_measured_K": round(k_decl / 18400, 2)},
           {"item": "moisture_sensitivity_per_wt_pct_at_18pct", "reduced_model_dlnP_dMC": round(-1.10 / 18.0, 4), "measured_law_dlnK_dMC": round(-alpha, 4),
            "ratio_reduced_to_measured": round((1.10 / 18.0) / alpha, 2)},
           {"item": "die_pressure_MPa_250_trials_fixed_recipe", "reduced_mean": round(float(np.mean(dp_red)), 3), "reduced_sd": round(float(np.std(dp_red)), 3),
            "literature_law_mean": round(float(np.mean(dp_lit)), 3), "literature_law_sd": round(float(np.std(dp_lit)), 3),
            "trials_over_12MPa_reduced": int(np.sum(np.array(dp_red) > 12.0)), "trials_over_12MPa_literature_law": int(np.sum(np.array(dp_lit) > 12.0))},
           {"item": "SME_kJ_kg_250_trials_fixed_recipe", "reduced_mean": round(float(np.mean(s_red)), 1), "reduced_sd": round(float(np.std(s_red)), 1),
            "literature_law_mean": round(float(np.mean(s_lit)), 1), "literature_law_sd": round(float(np.std(s_lit)), 1)},
           {"item": "feed_moisture_range_in_trials_pct_wb", "min": round(float(xs.min() * 100), 1), "max": round(float(xs.max() * 100), 1), "sd": round(float(xs.std() * 100), 2)}]
    keys = sorted({k for r in out for k in r}, key=lambda k: (k != "item", k))
    with open(HERE / "Ch11_Extrusion_Literature_Check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["evidence_status"]); w.writeheader()
        for r in out:
            r["evidence_status"] = "literature_rheology_one_starch_laboratory_rheometer_assumes_scenario_material_follows_the_law"
            w.writerow(r)
            print(r)


if __name__ == "__main__":
    main()
