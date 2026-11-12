"""Reduction checks for baking and extrusion (Ch11, Scenarios IIA-2 and IIB) and the cold chain (Ch12, Scenario IIIB).

These three scenarios are response surfaces with assigned constants. A resolved counterpart can be built only where the
package declares the equations and parameters needed:

  Extrusion  declared power-law rheology (K0 = 14200 Pa s^n, n = 0.34): at fixed throughput, geometry and exponent n the
             die pressure drop is proportional to the consistency index and the viscous-dissipation power is too. The assigned
             exponents on the consistency multiplier are compared with these analytical values.
  Baking     the package declares no oven temperature, product geometry or heat-transfer coefficient, so no resolved
             baking model can be constructed without adding assumptions. What can be computed is the surface-temperature
             sensitivity implied by the assigned acrylamide exponent and the declared activation energy (PD: 118 kJ/mol).
  Cold chain the reduced model holds the pallet-core temperature constant for the whole retail exposure. The check
             integrates the Arrhenius (P. fluorescens) and Ratkowsky (L. monocytogenes) growth over a sinusoidal core
             temperature of assigned amplitude A (sensitivity, not an estimate) with the same mean, and compares the
             executed L. monocytogenes growth scale with the declared cardinal parameters (mu_opt, T_min) under an assumed
             T_opt of 37 degC.

Scope: model-to-model and consistency checks. They are not validation against plant or distribution data.
Outputs (next to this script): Ch11_Extrusion_Reduction_Check.csv, Ch11_Baking_Reduction_Check.csv,
Ch12_ColdChain_Reduction_Check.csv.   Run: python verify_ch11_ch12_baking_extrusion_coldchain.py
"""
import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results"
R_GAS = 8.314462618


def load_params():
    with open(DATA / "kinetic_and_transport_parameters.csv", newline="", encoding="utf-8") as fh:
        return {(r["process_id"], r["symbol"]): float(r["value"]) for r in csv.DictReader(fh)}


def load_trials():
    with open(DATA / "monte_carlo_disturbances_seed42.csv", newline="", encoding="utf-8") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def write(name, rows):
    with open(HERE / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ================================================================== extrusion
def extrusion_check(P, trials):
    n = P[("Scenario_IIB_Extrusion", "n_extr")]
    p_max = P[("Scenario_IIB_Extrusion", "P_die_max_MPa")]
    rows = []
    for tier, gain in (("tier1_fixed_recipe", 0.00), ("tier2_open_loop_mechanistic", 0.35),
                       ("tier3_unconstrained_black_box_ml", 0.76), ("tier4_trustworthy_digital_twin", 0.94)):
        dp_red, dp_ana, sme_red, sme_ana, exc_red, exc_ana = [], [], [], [], 0, 0
        for tr in trials[:250]:
            k0, x = tr["extrusion_K0_mult"], tr["extrusion_X_feed_wb"]
            moist = (0.18 / x) ** (1.10 * (1.0 - 0.86 * gain))
            dp_r = 6.40 * (k0 ** (1.25 * (1.0 - 0.86 * gain))) * moist
            dp_a = 6.40 * (k0 ** (1.0 * (1.0 - 0.86 * gain))) * moist          # analytical power-law die: dP proportional to K
            sme_shift = (548.0 - 520.0) * (1.0 - gain) + (18.0 if gain == 0.35 else 0.0)
            moist_s = (0.18 / x) ** (0.82 * (1.0 - 0.88 * gain))
            s_r = (520.0 + sme_shift) * (k0 ** (1.0 - 0.88 * gain)) * moist_s
            s_a = (520.0 + sme_shift) * (k0 ** 1.0) * moist_s                   # viscous dissipation: power proportional to K
            dp_red.append(dp_r); dp_ana.append(dp_a); sme_red.append(s_r); sme_ana.append(s_a)
            exc_red += dp_r > p_max; exc_ana += dp_a > p_max
        rows.append({"tier": tier, "control_gain": gain, "n_power_law": n,
                     "dP_reduced_mean_MPa": round(np.mean(dp_red), 3), "dP_analytical_K_scaling_mean_MPa": round(np.mean(dp_ana), 3),
                     "dP_difference_mean_pct": round((np.mean(dp_red) / np.mean(dp_ana) - 1) * 100, 2),
                     "dP_sd_reduced": round(np.std(dp_red), 3), "dP_sd_analytical": round(np.std(dp_ana), 3),
                     "dP_over_interlock_reduced": exc_red, "dP_over_interlock_analytical": exc_ana,
                     "SME_reduced_mean_kJ_kg": round(np.mean(sme_red), 1), "SME_analytical_K_scaling_mean_kJ_kg": round(np.mean(sme_ana), 1),
                     "SME_sd_reduced": round(np.std(sme_red), 1), "SME_sd_analytical": round(np.std(sme_ana), 1),
                     "evidence_status": "consistency_check_of_assigned_exponents_against_declared_power_law_rheology"})
    return rows


# ================================================================== baking
def baking_check(P, trials):
    ea = P[("Scenario_IIA2_Baking", "Ea_AA_bake")]
    h = np.array([t["baking_H_oven_mult"] for t in trials[:250]])
    rows = []
    for gain in (0.0, 0.48, 0.78, 0.94):
        expo = 1.35 * (1.0 - 0.55 * gain)
        for Ts_c in (160.0, 180.0, 200.0):
            T = Ts_c + 273.15
            ds_per_lnh = expo / (ea / (R_GAS * T ** 2))               # K per unit ln(h): implied by the assigned exponent
            rows.append({"design_slot_gain": gain, "assumed_crust_surface_T_C": Ts_c, "assigned_exponent_AA_on_h_mult": round(expo, 3),
                         "declared_Ea_AA_J_mol": ea, "implied_dTs_per_unit_ln_h_K": round(ds_per_lnh, 1),
                         "implied_dTs_for_h_mult_sd_K": round(ds_per_lnh * float(np.std(np.log(h))), 2),
                         "h_mult_sd_log": round(float(np.std(np.log(h))), 3),
                         "evidence_status": "implied_sensitivity_only_no_resolved_baking_model_possible_from_declared_parameters"})
    return rows


# ================================================================== cold chain
POLICIES = [("1_Single_ReturnAir_Logger_Static_FIFO", 0.00, False, False, False),
            ("2_Peripheral_IoT_Surface_Arrhenius_S_FEFO", 0.48, False, False, False),
            ("3_18Zone_Zonal_ROM_Point_Estimate_FEFO", 0.88, False, False, True),
            ("4_Trustworthy_IVVO_DT_Algorithm_12_2", 0.945, True, True, True)]
THETA = np.linspace(0.0, 2 * math.pi, 721)[:-1]


def sso_factor(Tbar, A, ea):
    T = Tbar + A * np.sin(THETA)
    return float(np.mean(np.exp(-(ea / R_GAS) * (1.0 / (T + 273.15) - 1.0 / (Tbar + 273.15)))))


def lm_factor(Tbar, A):
    T = np.maximum(Tbar + A * np.sin(THETA), -1.72)
    return float(np.mean(((T + 1.72) / (8.0 + 1.72)) ** 2) / ((Tbar + 1.72) / (8.0 + 1.72)) ** 2)


def coldchain(P, trials, amp=0.0, lm_mode="executed"):
    mu_ref, ea = P[("Scenario_IIIB_ColdChain", "mu_ref_SSO")], P[("Scenario_IIIB_ColdChain", "Ea_SSO_cc")]
    ns_crit, nh_crit = P[("Scenario_IIIB_ColdChain", "Ns_crit_log")], P[("Scenario_IIIB_ColdChain", "Nh_crit_log")]
    mu_opt, t_min = P[("Scenario_IIIB_ColdChain", "mu_opt_Lm")], P[("Scenario_IIIB_ColdChain", "T_min_Lm_cc")]
    N = int(P[("Scenario_IIIB_ColdChain", "N_pallets_cc")])
    out = {}
    for name, rom, hsi, conf, _vfd in POLICIES:
        spoil = fp = unsafe = 0
        rsl_true_all = []
        for idx, p in enumerate(trials[:N]):
            dT = 3.8 + max(-0.8, p["coldchain_T_amb_offset_C"] * 0.95)
            apron = 1.65 if (idx % 8 == 0) else 0.0
            Tt = 2.1 + 0.55 * dT + apron
            Te = 2.1 + rom * (0.55 * dT) + (0.95 if hsi else 0.25 * rom) * apron
            N0s, N0l = p["coldchain_N0_SSO_log10"], p["coldchain_N0_Lm_log10"]
            mu_t = (mu_ref / math.log(10.0)) * math.exp(-(ea / R_GAS) * (1.0 / (Tt + 273.15) - 1.0 / 277.15))
            mu_e = (mu_ref / math.log(10.0)) * math.exp(-(ea / R_GAS) * (1.0 / (Te + 273.15) - 1.0 / 277.15))
            if amp > 0:
                mu_t *= sso_factor(Tt, amp, ea)
            N0e = (N0s + 0.04 * (N0s - 3.20)) if hsi else (3.20 + 0.22 * (N0s - 3.20))
            rsl_t = (ns_crit - N0s) / mu_t
            rsl_e = (ns_crit - N0e) / mu_e
            q = 14.5 if conf else (18.5 if rom == 0.48 else (9.5 if rom > 0.8 else 0.5))
            lb = rsl_e - q
            reff = 0.81 if conf else (0.72 if rom > 0.8 else (0.45 if rom > 0.3 else 0.0))
            t_req = 146.0 - reff * max(-30.0, min(62.0, (162.0 - rsl_t) * 0.92))
            spoil += (N0s + mu_t * t_req) > ns_crit
            if lb < 158.0 and rsl_t >= 165.0 and (not conf or idx == 19):
                fp += 1
            if lm_mode == "executed":
                mu_l = 0.0175 * (((Tt + 1.72) / 9.72) ** 2); mu_le = 0.0175 * (((Te + 1.72) / 9.72) ** 2)
            else:                                           # declared cardinal parameters, assumed T_opt = 37 degC
                mu_l = mu_opt * (max(Tt - t_min, 0.0) / (37.0 - t_min)) ** 2; mu_le = mu_opt * (max(Te - t_min, 0.0) / (37.0 - t_min)) ** 2
            if amp > 0:
                mu_l *= lm_factor(Tt, amp)
            t_lm = 0.0 if (conf and (N0l + (mu_le * 1.12) * t_req > nh_crit - 0.08)) else t_req
            unsafe += (N0l + mu_l * t_lm) > nh_crit
            rsl_true_all.append(rsl_t)
        out[name] = (spoil / N * 100, fp / N * 100, unsafe / N * 100, float(np.mean(rsl_true_all)))
    return out


def coldchain_check(P, trials):
    stored = {}
    with open(RESULTS / "Ch12_Table_12_4_Scenario_IIIB_ColdChain_Benchmark.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            stored[r["dispatch_policy"]] = r
    base = coldchain(P, trials)
    rows = []
    cases = [("reduced_constant_temperature", 0.0, "executed")] + [(f"temperature_amplitude_{a}C", a, "executed") for a in (0.5, 1.0, 2.0)] \
            + [("declared_cardinal_Listeria_Topt37", 0.0, "declared")]
    for label, amp, mode in cases:
        res = coldchain(P, trials, amp, mode)
        for name, (sp, fp, un, rsl) in res.items():
            rows.append({"case": label, "policy": name, "spoilage_pct": round(sp, 2), "false_positive_pct": round(fp, 2),
                         "unsafe_Lm_pct": round(un, 2), "mean_true_RSL_h": round(rsl, 1),
                         "spoilage_change_vs_reduced_points": round(sp - base[name][0], 2), "unsafe_change_vs_reduced_points": round(un - base[name][2], 2),
                         "matches_stored_csv": (round(sp, 2) == float(stored[name]["spoilage_waste_at_retail_pct"]) and
                                                round(un, 2) == float(stored[name]["unsafe_pallets_reaching_retail_Lm_gt_100_CFUg_pct"])) if label == "reduced_constant_temperature" else "",
                         "evidence_status": "model_to_model_sensitivity_amplitude_and_Topt_are_assigned_not_estimated"})
    return rows


def main():
    P, trials = load_params(), load_trials()
    for name, rows in (("Ch11_Extrusion_Reduction_Check.csv", extrusion_check(P, trials)),
                       ("Ch11_Baking_Reduction_Check.csv", baking_check(P, trials)),
                       ("Ch12_ColdChain_Reduction_Check.csv", coldchain_check(P, trials))):
        write(name, rows)
        print(name)
        for r in rows:
            print("  ", {k: v for k, v in r.items() if k != "evidence_status"})


if __name__ == "__main__":
    main()
