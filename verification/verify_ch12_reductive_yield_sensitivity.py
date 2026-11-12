"""Sensitivity of the Chapter 12 fermentation scenario (Scenario IIIA, Table 12.3) to the assigned reductive yield.

The stored table results/Ch12_Table_12_3_Scenario_IIIA_Fermentation_Benchmark.csv is produced by
run_all_benchmarks.run_ch12_fermentation_and_cold_chain_benchmarks with the reductive (overflow) biomass yield
YXS_red_ferm = 0.05 g/g read from data/kinetic_and_transport_parameters.csv (PD-104; the value of Sonnleitner and
Kaeppeli, 1986, which closes the carbon balance of the reductive pathway with the coded ethanol yield of 0.48 g/g).
0.10 and 0.15 g/g violate carbon closure by about 6 % and 12 % (106 % and 112 % carbon recovery) and are run as
sensitivities only; 0.15 g/g was the declared value until 11 November 2026. This script reruns the unchanged routine
in memory for the declared value and for 0.10 and 0.15 g/g and writes the endpoint rows of Table 12.3. Nothing in
results/ is overwritten (the routine's CSV writer is replaced by a capture).

Scope: executed sensitivity of an assigned input of a constructed scenario; not a model-to-model verification and not
validation against bioreactor data.

Output (next to this script): Ch12_Fermentation_Yield_Sensitivity.csv.
Run from this folder: python verify_ch12_reductive_yield_sensitivity.py
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_all_benchmarks as r  # noqa: E402

TABLE = "Ch12_Table_12_3_Scenario_IIIA_Fermentation_Benchmark.csv"
KEY = ("Scenario_IIIA_Fermentation", "YXS_red_ferm")
FIELDS = ["final_biomass_C_X_g_per_L_mean", "final_biomass_C_X_sd", "effective_yield_YXS_g_g_mean",
          "effective_yield_YXS_sd", "cumulative_ethanol_overflow_g_per_L_mean", "ethanol_sd",
          "DO_starvation_violations_pct", "cooling_jacket_saturation_events_pct",
          "specific_electrical_energy_kWh_per_kg_X", "energy_sd"]


def main():
    captured = {}
    r.write_csv = lambda filename, fieldnames, rows: captured.__setitem__(filename, rows)
    params = r.load_parameters()
    trials = r.load_mc_trials()
    declared = params[KEY]["value"]
    with open(HERE.parent / "results" / TABLE, newline="", encoding="utf-8") as fh:
        stored = {row["control_strategy"]: row for row in csv.DictReader(fh)}
    out = []
    for value in (declared, 0.10, 0.15):
        params[KEY]["value"] = value
        captured.clear()
        r.run_ch12_fermentation_and_cold_chain_benchmarks(params, trials)
        for row in captured[TABLE]:
            rec = {"YXS_red_g_g": value, "control_strategy": row["control_strategy"]}
            rec.update({f: row[f] for f in FIELDS})
            if value == declared:
                rec["matches_stored_csv"] = all(str(row[f]) == stored[row["control_strategy"]][f] for f in FIELDS)
            else:
                rec["matches_stored_csv"] = ""
            rec["evidence_status"] = "executed_sensitivity_of_assigned_input_not_validation"
            out.append(rec)
    params[KEY]["value"] = declared
    with open(HERE / "Ch12_Fermentation_Yield_Sensitivity.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for rec in out:
        print(rec["YXS_red_g_g"], rec["control_strategy"], rec["final_biomass_C_X_g_per_L_mean"],
              rec["effective_yield_YXS_g_g_mean"], rec["matches_stored_csv"])


if __name__ == "__main__":
    main()
