# Computational illustrations for *Digital Twins for Food Processing*

Companion package of the monograph:
code, inputs, stored results and verification scripts for the computational examples of Chapters 2–13. Paths of the form `reproducibility/...` in the book refer to the root of this repository.

**Status.** The package contains limiting-equation verification, reduced-model simulations, constructed response curves, assigned quantities and conditional accounting scenarios. It contains no industrial validation data set and no qualified food-process controller. Matching a stored file or a hash shows consistency with an artefact; it does not validate a model.

- Release accompanying the delivered text: tag `manuscript-2026-11-12` (Appendix C of the book).
- Licence: MIT (`LICENSE`) for code and files created for the book. The two files in `data/external` come from open-licence papers (CC BY 4.0, CC BY 3.0) and keep their licences; data of publications without an open licence are not included in this release (see `data/external/README.md`).
- Citation: `CITATION.cff`.

## Quick start

```
pip install -r requirements.txt          # Python 3.11, CPU only
pytest -q tests/test_package.py          # stored artefacts, regression references, manifest hashes
python tests/fast_rerun.py               # re-runs all modules except Chapter 9, in a temporary copy (5–10 min)
python run_all_benchmarks.py             # full run, including the 24-min Chapter 9 comparison
python run_all_benchmarks.py --manifest-only   # recomputes nothing: checks stored results, rewrites the manifest
```

The runner writes 24 result CSVs to `results/` and checks each against its reference in `benchmarks/`. Nine of them and the 14 figures are written by a figure generator that is not part of this repository; without it the runner skips them and says so. The `manifest_mode` field of `results/reproducibility_manifest.json` records which mode wrote the manifest.

## Layout

| Path | Content |
|---|---|
| `run_all_benchmarks.py` | scenario runners, regression checks, manifest writer |
| `data/` | the 131 parameter entries, their provenance register, the 500 seeded disturbances, the claim–evidence register, `DATA_DICTIONARY.md` |
| `results/`, `benchmarks/` | 24 stored result CSVs and their independently derived regression references |
| `verification/` | executed checks and their output CSVs (run separately, see below) |
| `evidence_register.json`, `results/reproducibility_manifest.json` | evidence metadata, SHA-256 hashes |
| `tests/`, `.github/workflows/ci.yml` | tests and continuous integration (Ubuntu, Python 3.11) |

Every result CSV has an `evidence_status` column. The labels used in the book are: *executed benchmark* (code run, results regenerated), *model-to-model verification* (reduced against resolved model), *assigned scenario* (programmed response, not a computation of a physical process) and *validation against independent observations* (comparison with measured data). Legacy row identifiers in some CSV files name rows only and do not describe executed methods.

## Verification scripts

Run from `verification/` with `verification/requirements.txt`. None is part of `run_all_benchmarks.py`.

| Script | Output | Time |
|---|---|---|
| `verify_ch10_reduction_error.py` | `Ch10_Reduction_Error_*.csv`: three-node retort model against a resolved conduction model, all policies of Table 10.3 on the same 500 trials | 27 min |
| `verify_ch10_ch11_ch12_reduction_error.py` | UHT, drying and fermentation reduction checks | 3 min |
| `verify_ch11_ch12_baking_extrusion_coldchain.py` | baking, extrusion, cold-chain reduction checks | 1 min |
| `verify_ch11_baking_resolved.py` | conduction-only loaf model under two oven programmes | 2 min |
| `verify_ch11_baking_physical.py` | calibration and verification on one published loaf | 4 min |
| `verify_ch11_extrusion_literature.py`, `verify_ch10_ch11_industrial_pilot_data.py`, `verify_ch11_ch12_literature_remaining_items.py` | comparison with published values | 1 min each |
| `verify_ch12_reductive_yield_sensitivity.py` | fermentation yield sensitivity | 10 s |
| `benchmark_ch03_dqi_gate.py`, `benchmark_ch05_observer_example.py`, `benchmark_ch09_margin_ablation.py` | executed tests of Chapters 3, 5 and 9 (Tables 3.7, 5.7, 9.8) | 1–12 min |
| `benchmark_ch04_drying_architectures.py` | seven architectures on a drying generator (Tables 4.6–4.7); needs PyTorch and scikit-learn | hours |
| `extract_apple_drying_data.py`, `benchmark_ch04_physical_drying.py`, `check_ch04_physical_endtime.py`, `check_ch04_ude_factor.py` | retrospective reanalysis of published apple-drying curves (Table 4.8) and its checks | minutes to hours |

Notes on specific results:

- **Chapter 10.** Binomial (Clopper–Pearson) bounds in `Ch10_Reduction_Error_Summary.csv` are given only for policies that were not tuned on the 500 trials. For policies whose hold time, pulse end or margin was tuned on the same trials the count is in-sample, and the bound columns read `not_applicable_in_sample` (column `interval_status`). Two-fold cross-fitting is a consistency check of the tuning, not an independent test.
- **Chapter 9.** The closed-loop comparison (Tables 9.6–9.7) uses 200 seeded realisations of a reduced pasteuriser model; computation-time rows are wall-clock and excluded from numerical checks, and solver diagnostics depend on the numerical library build. Results are not validation of any controller.
- **Chapter 4, Table 4.8.** The reanalysis is retrospective: the thickness input includes the final measurement and some models use the end time of the test curve, so the ranking is not a prospective comparison.
- **Seeds and versions.** Seeds are fixed in the scripts (42 for the Monte Carlo table). Stored files were produced with Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1 and, for Chapter 4, scikit-learn 1.9.1 and PyTorch 2.15.0 (CPU); results may differ in the last digits on other versions, more for the neural and Gaussian-process models.

## Data

- `data/kinetic_and_transport_parameters.csv`: 131 entries, each with a stable design ID, value, unit and origin in `data/parameter_provenance_register.csv` (124 author-selected inputs, 6 arithmetic derivations, 1 literature-reported value). Origin is not calibration for an actual product.
- `data/external/`: apple-drying curves digitised from Dajbych et al. (CC BY 4.0) and air temperatures of a biscuit oven read from Mayo Bayón et al. (CC BY 3.0). Four verification scripts need further files from publications without an open licence; they are not included in this release, the scripts stop with a message when a file is missing, and `data/external/README.md` lists the expected columns.
- A few constants coded in `run_all_benchmarks.py` are not among the 131 entries (oxygen requirement and ethanol inhibition constant of the fermentation routine, Listeria growth constants of Tables 12.4–12.5); they are assigned scenario values and are documented in `evidence_register.json`.

## Figures

The 14 figures are not part of the public repository; they are supplied to the publisher with the manuscript. Their SHA-256 values are in the manifest. Figures 10.1(b) and 11.1 label their categories as assigned scenarios and design slots, not as executed controllers or models. Figures differ bitwise between Matplotlib versions.

## Independent re-execution

- *31 October 2026*: the package, copied without results and figures into a new environment built from `requirements.txt`, was run end to end. 21 of the 22 result CSVs then present were byte-identical; the Chapter 9 comparison agreed in every excursion count and in its physical quantities to within 0.2 %; all 14 figures were pixel-identical. The nine Chapter 10–12 verification scripts regenerated 16 CSV files byte for byte (`Ch10_Reduction_Error_Trials.csv` differed by at most 1.1e-7 relative).
- *9, 11 and 12 November 2026*: all modules except Chapter 9 were re-run in a temporary copy of the package (`tests/fast_rerun.py`, with the figure generator); the result CSVs passed their regression checks and those written by `run_all_benchmarks.py` were byte-identical after LF canonicalisation, apart from the intended changes. On 12 November 2026 `verify_ch10_reduction_error.py` was also re-run: only the bound columns of the tuned policies changed.
- The Chapter 4 scripts needing PyTorch and the PDF extraction were not repeated in a new environment.

## Tests, line endings and hashes

`.gitattributes` disables end-of-line conversion (`* -text`); files are stored byte for byte. Manifest hashes of `.csv`, `.json` and `.py` files are taken after converting CRLF and lone CR to LF (`sha256_artefact()` in the runner and `sha()` in `tests/test_package.py`); `.png` and `.docx` files are hashed as bytes. Figure and generator hashes are checked only when those files are present locally.
