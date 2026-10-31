"""Resolved check of the baking scenario (Chapter 11, Scenario IIA-2) with stated oven programmes.

Oven programmes (supplied for this check as typical bread-baking practice):
  artisan   white or sourdough bread: preheated oven at 220-250 degC, lowered to 200-220 degC after the first 10-20 min;
            here 240 degC for the first 15 min, then 210 degC.
  wholemeal wholemeal or dark-flour bread: a moderate 160-170 degC throughout; here 165 degC.

Resolved model: one-dimensional transient conduction through the half-thickness of a hearth loaf (42 mm, text of Ch11;
symmetry plane at the centre) with a convective boundary to the oven, and local evaporation handled by an apparent heat
capacity (latent heat released over 99-101 degC for the evaporable water of each cell). There is no moisture transport.

Declared inputs: half-thickness 42 mm, initial dry-basis moisture 0.78 kg/kg and nominal residence time 22 min (text of
Ch11); the minimum core gelatinisation temperature 96 degC and the acrylamide activation energy 118 kJ/mol (parameter table).
Assumed inputs (typical values, stated as such): thermal conductivity 0.35 W/(m K), density 800 kg/m3, specific heat
3000 J/(kg K), initial temperature 30 degC, evaporable fraction of the water 0.25 (a bake loss of about 11 % of the dough mass),
and the heat-transfer coefficient h (15, 25 and 40 W/(m2 K) as a sensitivity).

Compared with the reduced model (run_all_benchmarks.py): (i) the reduced gelatinisation 1 - exp(-0.218 h_mult t) at t = 18.2 min
(about 98 %) against the time the resolved core needs to reach the declared 96 degC; (ii) the assigned exponent 1.35 of the acrylamide
response to the oven heat-transfer multiplier against the exponent of an Arrhenius surface integral (118 kJ/mol) from the resolved
surface temperature, evaluated on a +-15 % range of h around its nominal value. A conduction-only model has no vapour transport (heat-pipe
effect), so its core heating time is a conditional result of this model, not a proven bound on baking time; the check reports it and does not tune it.

Scope: a conduction-only resolved model with assumed properties; the comparison is not a validation against bake data.
Output: Ch11_Baking_Resolved_Check.csv.   Run: python verify_ch11_baking_resolved.py
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
R_GAS = 8.314462618
L_HALF, X0_DB, T_BAKE_MIN = 0.042, 0.78, 22.0
K_TH, RHO, CP, T_INIT, EVAP_FRAC, LV = 0.35, 800.0, 3000.0, 30.0, 0.25, 2.26e6
PROGRAMMES = {"artisan_240C_15min_then_210C": lambda t_min: 240.0 if t_min < 15.0 else 210.0,
              "wholemeal_165C": lambda t_min: 165.0}


def load_params():
    with open(DATA / "kinetic_and_transport_parameters.csv", newline="", encoding="utf-8") as fh:
        return {(r["process_id"], r["symbol"]): float(r["value"]) for r in csv.DictReader(fh)}


def load_trials():
    with open(DATA / "monte_carlo_disturbances_seed42.csv", newline="", encoding="utf-8") as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def bake(programme, h, t_end_min=60.0, n=42, dt=2.0, Ea=118000.0, k_mult=1.0):
    dx = L_HALF / n
    w_wet = X0_DB / (1.0 + X0_DB)                                     # water mass fraction of the dough
    latent = LV * w_wet * EVAP_FRAC * RHO                              # J/m3 released over 99-101 degC
    K = K_TH * k_mult
    G_surf = 1.0 / (1.0 / h + dx / (2.0 * K))                       # oven-to-surface-cell conductance
    T = np.full(n, T_INIT)
    t_core96 = None
    gel_I, aa_I, time_above = 0.0, 0.0, 0.0
    out = {}
    for step in range(int(t_end_min * 60 / dt)):
        t_min = step * dt / 60.0
        c_app = RHO * CP + np.where((T > 99.0) & (T < 101.0), latent / 2.0, 0.0)
        a = K / dx ** 2
        lower = np.full(n, -a); diag = np.full(n, 2 * a); upper = np.full(n, -a)
        diag[0] = a; lower[0] = 0.0                                    # symmetry plane
        rhs = c_app / dt * T
        diag[-1] = a + G_surf / dx
        rhs[-1] += G_surf / dx * PROGRAMMES[programme](t_min)
        ab = np.vstack([np.r_[0.0, upper[:-1]], diag + c_app / dt, np.r_[lower[1:], 0.0]])
        T = solve_banded((1, 1), ab, rhs)
        t_now = (step + 1) * dt / 60.0
        if t_core96 is None and T[0] >= 96.0:
            t_core96 = t_now
        if T[0] >= 96.0:
            time_above += dt / 60.0
        Ts = T[-1] + 273.15
        if T[-1] > 100.0:                                               # dry crust: acrylamide-forming temperature range
            aa_I += math.exp(-(Ea / R_GAS) * (1.0 / Ts - 1.0 / 453.15)) * dt / 60.0
        if abs(t_now - T_BAKE_MIN) < dt / 120.0:
            out = {"core_T_at_22min": T[0], "surface_T_at_22min": T[-1], "aa_index_22min": aa_I}
    return t_core96, out, aa_I


def core_time(prog, h, k_mult):
    return bake(prog, h, t_end_min=240.0, k_mult=k_mult)[0]


def calibrate_k_mult(prog, h, target_min=18.2):
    """Conductivity multiplier (effective, standing in for vapour transport) for which the core reaches 96 degC at target_min."""
    lo, hi = 1.0, 400.0
    for _ in range(18):
        mid = math.sqrt(lo * hi)
        t = core_time(prog, h, mid)
        lo, hi = (mid, hi) if (t is None or t > target_min) else (lo, mid)
    return math.sqrt(lo * hi)


def aa_exponent(prog, h_nom, k_mult, Ea):
    grid = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    aa = np.array([bake(prog, h_nom * m, t_end_min=T_BAKE_MIN, Ea=Ea, k_mult=k_mult)[1].get("aa_index_22min", 0.0) for m in grid])
    return float(np.polyfit(np.log(grid), np.log(aa), 1)[0]) if np.all(aa > 0) else float("nan")


def main():
    P = load_params()
    Ea = P[("Scenario_IIA2_Baking", "Ea_AA_bake")]
    h_mult = np.array([t["baking_H_oven_mult"] for t in load_trials()[:250]])
    alpha_red = float(np.mean((1.0 - np.exp(-0.218 * h_mult * 18.2)) * 100.0))
    rows = []
    for prog in PROGRAMMES:
        for h_nom in (15.0, 25.0, 40.0):
            t96 = core_time(prog, h_nom, 1.0)
            _, o, _ = bake(prog, h_nom, t_end_min=T_BAKE_MIN, Ea=Ea)
            e1 = aa_exponent(prog, h_nom, 1.0, Ea)
            rows.append({"programme": prog, "h_nominal_W_m2K": h_nom,
                         "conduction_only_core_T_at_22min_C": round(float(o["core_T_at_22min"]), 1),
                         "conduction_only_time_to_core_96C_min": ("not reached in 240 min" if t96 is None else round(t96, 1)),
                         "reduced_gelatinisation_at_18p2min_pct": round(alpha_red, 1),
                         "exponent_acrylamide_on_h_conduction_only": (round(e1, 2) if e1 == e1 else "no dry crust"),
                         "assigned_exponent_acrylamide_on_h": 1.35,
                         "evidence_status": "model_to_model_conduction_only_assumed_properties_and_h_not_validation"})
            print({k: v for k, v in rows[-1].items() if k != "evidence_status"}, flush=True)
    with open(HERE / "Ch11_Baking_Resolved_Check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
