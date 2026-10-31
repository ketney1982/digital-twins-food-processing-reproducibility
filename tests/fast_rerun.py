"""Re-run every computational module except the 24-minute Chapter 9 closed-loop comparison, in a temporary copy of the
package (the repository files are not touched), and fail if any result differs from its regression reference.

Usage (from the repository root): python tests/fast_rerun.py      (about 5-10 minutes)
The full run, including Chapter 9, is `python run_all_benchmarks.py`.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER = '''
import sys
import run_all_benchmarks as r
p = r.load_parameters(); mc = r.load_mc_trials()
r.run_foundational_benchmarks(p, mc)
m7 = r.run_ch10_thermal_benchmarks(p, mc)
m8a = r.run_ch11_dehydration_baking_extrusion_benchmarks(p, mc)
m8b = r.run_ch12_fermentation_and_cold_chain_benchmarks(p, mc)
r.run_ch13_economic_and_lca_benchmarks(p, mc, m7, m8a, m8b)
from generate_expanded_benchmarks_and_figures import run_expanded_verification_benchmarks, generate_all_monograph_figures
run_expanded_verification_benchmarks(p, mc, r.write_csv, r.RES_DIR)
generate_all_monograph_figures(r.RES_DIR, r.BASE_DIR / "figures")
print("FAST RERUN OK")
'''

with tempfile.TemporaryDirectory() as tmp:
    work = pathlib.Path(tmp) / "reproducibility"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "__pycache__", "results", "figures", "tests"))
    (work / "results").mkdir()
    (work / "figures").mkdir()
    res = subprocess.run([sys.executable, "-c", DRIVER], cwd=work)
    sys.exit(res.returncode)
