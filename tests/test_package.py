"""Fast, platform-independent checks of the published package (run from the repository root: pytest -q).

They check the stored artefacts, not the science: that the inputs load, that every stored result file agrees with its
regression reference, that the SHA-256 values of the manifest hold for the files of this checkout, that the evidence
register validates, and that the Chapter 9 regression check is sensitive to a change of an excursion proportion or of
a steam value.  Re-running the computations is done by tests/fast_rerun.py (all modules except the 24-minute Chapter 9
comparison) and by `python run_all_benchmarks.py` (everything).
"""
import hashlib
import json
import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_all_benchmarks as r  # noqa: E402

MANIFEST = json.loads((ROOT / "results" / "reproducibility_manifest.json").read_text(encoding="utf-8"))


def sha(path):
    path = pathlib.Path(path)
    content = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".py"}:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def test_inputs_load():
    assert len(r.load_parameters()) == 131
    assert len(r.load_mc_trials()) == 500


def test_every_result_has_a_reference_and_agrees():
    names = sorted(p.name for p in (ROOT / "results").glob("*.csv"))
    assert len(names) == 24  # 23 until 11 November 2026; Ch06_Zone_Temperature_Series.csv was added
    for n in names:
        ref = ROOT / "benchmarks" / n
        assert ref.is_file(), n
        if n == r.CH09_CTRL_CSV:
            r.ch09_regression_check(ROOT / "results" / n, ref)
        else:
            r.verify_result(ROOT / "results" / n, ref)


def test_manifest_hashes_hold_for_this_checkout():
    for k, v in MANIFEST["outputs"].items():
        assert sha(ROOT / "results" / k) == v, k
    for k, v in MANIFEST["reference_hashes"].items():
        assert sha(ROOT / "benchmarks" / k) == v, k
    for k, v in MANIFEST["figure_hashes"].items():
        figure = ROOT / "figures" / k
        if figure.is_file():
            assert sha(figure) == v, k
    assert sha(ROOT / "run_all_benchmarks.py") == MANIFEST["script_sha256"]
    generator = ROOT / "generate_expanded_benchmarks_and_figures.py"
    if generator.is_file():
        assert sha(generator) == MANIFEST["expanded_generator_sha256"]
    assert sha(ROOT / "evidence_register.json") == MANIFEST["evidence_register_sha256"]
    for k, v in MANIFEST["inputs"].items():
        assert sha(ROOT / "data" / k) == v, k


def test_fourteen_figures():
    figures = list((ROOT / "figures").glob("*.png"))
    if not figures:
        pytest.skip("Figure files are intentionally excluded from the repository")
    assert len(figures) == 14


def test_evidence_register_validates():
    out = r.validate_evidence_register()
    assert out["parameter_records"] == 131


def test_verification_outputs_are_non_empty():
    files = sorted((ROOT / "verification").glob("*.csv"))
    assert len(files) >= 24
    for f in files:
        assert len(pd.read_csv(f)) > 0, f.name


def test_ch09_regression_check_is_sensitive(tmp_path):
    name = r.CH09_CTRL_CSV
    ref = ROOT / "benchmarks" / name
    stored = pd.read_csv(ROOT / "results" / name)
    for metric, factor_col, change in [("runs_with_critical_limit_excursion", "estimate", lambda x: x + 0.5),
                                       ("steam_energy", "estimate", lambda x: x * 1.02)]:
        d = stored.copy()
        i = d.index[d.metric == metric][3]
        d.loc[i, factor_col] = change(d.loc[i, factor_col])
        out = tmp_path / (metric + ".csv")
        d.to_csv(out, index=False)
        with pytest.raises(AssertionError):
            r.ch09_regression_check(out, ref)
