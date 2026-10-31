"""Comparison of seven architectures on published physical drying curves (Chapter 4).

Data: Red Delicious apple slices dried in a hot-air oven at 40, 50, 60, 70 and 80 degC, duplicate tests I and II
(Processes 2026, 14(12), 2015, https://doi.org/10.3390/pr14122015, CC BY 4.0), digitised from the vector figures of the
paper by extract_apple_drying_data.py and checked against the ten published slopes of its Table 2 (largest deviation 2.04 %).
Target: moisture ratio MR(t) as defined in the paper (moisture content relative to the final mass, divided by the
initial value). Permitted inputs: time, air temperature and the measured average half-thickness (Table 1).

Architectures: M1 Fickian slab series with two calibrated parameters (ln D0, Ea); M1a the same series with the
declared descriptors of the book (PD-081, PD-082), not fitted; M2 ridge polynomial; M3 MLP; M4 GPR;
M5 parallel hybrid (M1 + GPR residual); M6 UDE (lumped first-order term with a neural correction); M7 LSTM.
Only training curves are used for fitting and for hyperparameter selection (curve-grouped cross-validation).

Splits (whole curves are withheld): replicate transfer (train on test I, test on test II, and the reverse),
temperature extrapolation (train 40-70 degC, test 80 degC; train 50-80 degC, test 40 degC) and temperature
interpolation (leave out 50, 60 or 70 degC in both tests).

Sensitivity of the ranking (variant column of the output files): "nominal_seed0" reproduces Table 4.8 (formerly 4.7);
"nominal_seed1" to "nominal_seed4" repeat the stochastic learners (M3, M6, M7) with other training seeds.
With --budget-variants, "budget_low" and "budget_high" also change the training budget of M3, M4/M5 (optimiser
restarts), M6 and M7 as in benchmark_ch04_drying_architectures.py (not run for the stored files). Models that are not
refitted are copied from nominal_seed0 so that each variant ranks all eight models.

Scope: ten curves from one laboratory and one product. Results describe prediction of these published curves; they are
not an estimate of performance in an industrial dryer.
Outputs (next to this script): Ch04_Physical_Drying_Digitisation_Check.csv, Ch04_Physical_Drying_Folds.csv,
Ch04_Physical_Drying_Summary.csv.     Run: python benchmark_ch04_physical_drying.py [--workers N]
"""
import argparse
import csv
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch  # must be imported before numpy/scipy/sklearn on Windows (DLL load order)
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares
from scipy.stats import rankdata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "external" / "apple_drying_curves_Processes2026.csv"
R_GAS = 8.314462618
D0_BOOK, EA_BOOK = 4.2e-6, 28400.0                      # PD-081, PD-082
L_AVG = {("I", 40): 0.00324, ("I", 50): 0.00297, ("I", 60): 0.00322, ("I", 70): 0.00324, ("I", 80): 0.00343,
         ("II", 40): 0.00317, ("II", 50): 0.00320, ("II", 60): 0.00325, ("II", 70): 0.00320, ("II", 80): 0.00332}
PAPER_SLOPES = {("I", 40): -5.399, ("I", 50): -7.337, ("I", 60): -10.685, ("I", 70): -11.985, ("I", 80): -15.169,
                ("II", 40): -6.239, ("II", 50): -6.590, ("II", 60): -8.254, ("II", 70): -11.081, ("II", 80): -14.295}
T_EVAL = np.arange(10.0, 521.0, 10.0)                   # evaluation grid (min), common to all curves
BUDGETS = {
    "nominal": {"mlp_max_iter": 4000, "mlp_tol": 1e-4, "mlp_no_change": 10, "mlp_restarts": 3, "gpr_restarts": 1,
                "ude_nfev": 40, "lstm_epochs": 400, "lstm_seeds": 3},
    "low": {"mlp_max_iter": 10, "mlp_tol": 1e-4, "mlp_no_change": 10, "mlp_restarts": 1, "gpr_restarts": 0,
            "ude_nfev": 10, "lstm_epochs": 100, "lstm_seeds": 1},
    "high": {"mlp_max_iter": 4000, "mlp_tol": 1e-6, "mlp_no_change": 50, "mlp_restarts": 5, "gpr_restarts": 4,
             "ude_nfev": 120, "lstm_epochs": 1600, "lstm_seeds": 5},
}
# variant: (budget, training seed, models refitted; the others are copied from nominal_seed0)
VARIANTS = {"nominal_seed0": ("nominal", 0, None)}
VARIANTS.update({f"nominal_seed{k}": ("nominal", 100 * k, ("M3", "M6", "M7")) for k in range(1, 5)})
BUDGET_VARIANTS = {"budget_low": ("low", 0, ("M3", "M4", "M5", "M6", "M7")), "budget_high": ("high", 0, ("M3", "M4", "M5", "M6", "M7"))}


def load_curves():
    raw = {}
    with open(DATA, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            raw.setdefault((r["test"], int(r["temperature_C"])), []).append((float(r["time_min"]), float(r["moisture_ratio"])))
    curves = []
    for (test, T), pts in sorted(raw.items()):
        t = np.array([p[0] for p in pts]); m = np.array([p[1] for p in pts])
        curves.append({"id": f"{test}-{T}", "test": test, "T": T, "L": L_AVG[(test, T)], "t_end": float(t.max()),
                       "mr": np.interp(T_EVAL, t, m)})
    return curves


def digitisation_check(curves):
    rows = []
    raw = {}
    with open(DATA, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            raw.setdefault((r["test"], int(r["temperature_C"])), []).append((float(r["time_min"]), float(r["moisture_ratio"])))
    for key, pts in sorted(raw.items()):
        t = np.array([p[0] for p in pts]) * 60.0; m = np.array([p[1] for p in pts]); sel = m >= 0.3
        slope = float(np.polyfit(t[sel], np.log(m[sel]), 1)[0]) * 1e5
        rows.append({"test": key[0], "temperature_C": key[1], "slope_digitised_1e-5_per_s": round(slope, 3),
                     "slope_published_1e-5_per_s": PAPER_SLOPES[key],
                     "relative_difference_pct": round((slope / PAPER_SLOPES[key] - 1) * 100, 2)})
    return rows


# ------------------------------------------------------------------------------ models
def fick_mr(t_min, T_c, L, t_end, ln_d0, ea):
    D = math.exp(ln_d0) * math.exp(-ea / (R_GAS * (T_c + 273.15)))
    n = np.arange(60)[:, None]

    def S(tm):
        tau = (2 * n + 1) ** 2 * math.pi ** 2 * D * (np.atleast_1d(tm)[None, :] * 60.0) / (4 * L ** 2)
        return (8 / math.pi ** 2) * np.sum(np.exp(-tau) / (2 * n + 1) ** 2, axis=0)
    s_end = float(S(t_end)[0])
    return (S(t_min) - s_end) / (1.0 - s_end)


def feats(curves):
    return np.array([[t / 60.0, c["T"], c["L"] * 1000.0] for c in curves for t in T_EVAL])


def targ(curves):
    return np.concatenate([c["mr"] for c in curves])


def grp(curves):
    return np.repeat(np.arange(len(curves)), len(T_EVAL))


class Fick:
    name = "M1_Mechanistic_Fickian_series"

    def fit(self, tr):
        def res(p):
            return np.concatenate([fick_mr(T_EVAL, c["T"], c["L"], c["t_end"], p[0], p[1]) - c["mr"] for c in tr])
        self.p = least_squares(res, [math.log(1e-6), 25000.0], x_scale=[1.0, 1e4]).x
        return self

    def predict(self, te):
        return np.concatenate([fick_mr(T_EVAL, c["T"], c["L"], c["t_end"], *self.p) for c in te])


class FickBook(Fick):
    name = "M1a_Declared_descriptors_not_fitted"

    def fit(self, tr):
        self.p = [math.log(D0_BOOK), EA_BOOK]
        return self


class Poly:
    name = "M2_Polynomial_ridge"

    def fit(self, tr):
        X, y, g = feats(tr), targ(tr), grp(tr); best = None
        for deg in (2, 3):
            for a in (1e-2, 1.0, 100.0):
                err = []
                for i, j in GroupKFold(min(4, len(tr))).split(X, y, g):
                    m = make_pipeline(StandardScaler(), PolynomialFeatures(deg), Ridge(alpha=a)).fit(X[i], y[i])
                    err.append(np.mean((m.predict(X[j]) - y[j]) ** 2))
                if best is None or np.mean(err) < best[0]:
                    best = (np.mean(err), deg, a)
        self.m = make_pipeline(StandardScaler(), PolynomialFeatures(best[1]), Ridge(alpha=best[2])).fit(X, y)
        return self

    def predict(self, te):
        return self.m.predict(feats(te))


class MLP:
    name = "M3_MLP"

    def __init__(self, budget=None):
        self.b = budget or BUDGETS["nominal"]

    def _mk(self, h, a, s):
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, activation="tanh", alpha=a, max_iter=self.b["mlp_max_iter"],
                                                            tol=self.b["mlp_tol"], n_iter_no_change=self.b["mlp_no_change"], random_state=s))

    def fit(self, tr, seed=0):
        X, y, g = feats(tr), targ(tr), grp(tr); best = None
        for h in ((16,), (32, 32)):
            for a in (1e-4, 1e-2):
                err = []
                for i, j in GroupKFold(min(4, len(tr))).split(X, y, g):
                    err.append(np.mean((self._mk(h, a, seed).fit(X[i], y[i]).predict(X[j]) - y[j]) ** 2))
                if best is None or np.mean(err) < best[0]:
                    best = (np.mean(err), h, a)
        self.ms = [self._mk(best[1], best[2], seed + s).fit(X, y) for s in range(self.b["mlp_restarts"])]
        return self

    def predict(self, te):
        return np.mean([m.predict(feats(te)) for m in self.ms], axis=0)


def _gpr(d, n_restarts=1):
    k = ConstantKernel(0.5) * Matern(np.ones(d), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=n_restarts, random_state=0))


class GPR:
    name = "M4_GPR"

    def __init__(self, budget=None):
        self.b = budget or BUDGETS["nominal"]

    def fit(self, tr):
        self.m = _gpr(3, self.b["gpr_restarts"]).fit(feats(tr), targ(tr))
        return self

    def predict(self, te):
        return self.m.predict(feats(te))


class Hybrid:
    name = "M5_Parallel_hybrid_M1_plus_GPR_residual"

    def __init__(self, budget=None):
        self.b = budget or BUDGETS["nominal"]

    def fit(self, tr, mech):
        self.mech = mech
        base = mech.predict(tr)
        self.m = _gpr(4, self.b["gpr_restarts"]).fit(np.c_[feats(tr), base], targ(tr) - base)
        return self

    def predict(self, te):
        base = self.mech.predict(te)
        return base + self.m.predict(np.c_[feats(te), base])


NH = 6


def ude_roll(p, curves):
    lnk0, ea = p[0], p[1]
    w1 = p[2:2 + 3 * NH].reshape(NH, 3); b1 = p[2 + 3 * NH:2 + 4 * NH]; w2 = p[2 + 4 * NH:2 + 5 * NH]; b2 = p[2 + 5 * NH]
    T = np.array([c["T"] for c in curves]); L = np.array([c["L"] for c in curves]) * 1000.0
    k = np.exp(lnk0) * np.exp(-ea / (R_GAS * (T + 273.15)))              # 1/min
    ts, ls = (T - 60.0) / 15.0, (L - 3.2) / 0.1
    dt = 2.0; X = np.ones(len(curves)); out = []
    step_per = int(10 / dt)
    for s in range(1, int(T_EVAL[-1] / dt) + 1):
        def f(x):
            z = np.stack([x, ts, ls], axis=1)
            return -k * (1.0 + np.tanh(z @ w1.T + b1) @ w2 + b2) * x
        k1 = f(X); k2 = f(X + 0.5 * dt * k1); k3 = f(X + 0.5 * dt * k2); k4 = f(X + dt * k3)
        X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s % step_per == 0:
            out.append(X.copy())
    return np.stack(out, axis=1)


class UDE:
    name = "M6_UDE_neural_correction"

    def __init__(self, budget=None):
        self.b = budget or BUDGETS["nominal"]

    def _fit(self, tr, lam, mech, rng):
        obs = np.stack([c["mr"] for c in tr]); Lm = np.mean([c["L"] for c in tr])
        lnk0 = mech.p[0] + math.log(math.pi ** 2 / (4 * Lm ** 2) * 60.0)
        p0 = np.r_[lnk0, mech.p[1], rng.normal(0, 0.1, 3 * NH), np.zeros(NH), rng.normal(0, 0.1, NH), 0.0]
        res = lambda p: np.r_[(ude_roll(p, tr) - obs).ravel(), math.sqrt(lam) * p[2:]]
        return least_squares(res, p0, x_scale=np.r_[1.0, 1e4, np.ones(len(p0) - 2)], max_nfev=self.b["ude_nfev"]).x

    def fit(self, tr, mech, seed=0):
        rng = np.random.default_rng(seed); idx = np.arange(len(tr)); best = None
        for lam in (1e-3, 1e-1):
            err = []
            for i, j in GroupKFold(min(4, len(tr))).split(idx, groups=idx):
                p = self._fit([tr[a] for a in i], lam, mech, rng)
                err.append(np.mean((ude_roll(p, [tr[a] for a in j]) - np.stack([tr[a]["mr"] for a in j])) ** 2))
            if best is None or np.mean(err) < best[0]:
                best = (np.mean(err), lam)
        self.p = self._fit(tr, best[1], mech, rng)
        return self

    def predict(self, te):
        return ude_roll(self.p, te).ravel()


class LSTM:
    name = "M7_LSTM"

    def __init__(self, budget=None):
        self.b = budget or BUDGETS["nominal"]

    def _seq(self, cs):
        return feats(cs).reshape(len(cs), len(T_EVAL), -1)

    def _train(self, X, Y, wd, seed):
        import torch
        torch.manual_seed(seed); torch.set_num_threads(1)

        class Net(torch.nn.Module):
            def __init__(s):
                super().__init__(); s.l = torch.nn.LSTM(3, 16, batch_first=True); s.o = torch.nn.Linear(16, 1)

            def forward(s, x):
                return s.o(s.l(x)[0]).squeeze(-1)
        net = Net(); opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=wd)
        X = torch.tensor(X, dtype=torch.float32); Y = torch.tensor(Y, dtype=torch.float32)
        for _ in range(self.b["lstm_epochs"]):
            opt.zero_grad(); loss = torch.mean((net(X) - Y) ** 2); loss.backward(); opt.step()
        return net

    def _pred(self, net, X):
        import torch
        with torch.no_grad():
            return net(torch.tensor(X, dtype=torch.float32)).numpy()

    def fit(self, tr, seed=0):
        Xs = self._seq(tr); flat = Xs.reshape(-1, 3)
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-9
        Xn = (Xs - self.mu) / self.sd; Y = np.stack([c["mr"] for c in tr]); idx = np.arange(len(tr)); best = None
        for wd in (1e-4, 1e-2):
            err = []
            for i, j in GroupKFold(min(4, len(tr))).split(idx, groups=idx):
                err.append(np.mean((self._pred(self._train(Xn[i], Y[i], wd, seed), Xn[j]) - Y[j]) ** 2))
            if best is None or np.mean(err) < best[0]:
                best = (np.mean(err), wd)
        self.nets = [self._train(Xn, Y, best[1], seed + s) for s in range(self.b["lstm_seeds"])]
        return self

    def predict(self, te):
        Xn = (self._seq(te) - self.mu) / self.sd
        return np.mean([self._pred(n, Xn) for n in self.nets], axis=0).ravel()


# ------------------------------------------------------------------------------ evaluation
def splits(curves):
    by = lambda f: [c for c in curves if f(c)]
    out = []
    out.append(("replicate_I_to_II", by(lambda c: c["test"] == "I"), by(lambda c: c["test"] == "II")))
    out.append(("replicate_II_to_I", by(lambda c: c["test"] == "II"), by(lambda c: c["test"] == "I")))
    out.append(("extrapolate_80C", by(lambda c: c["T"] <= 70), by(lambda c: c["T"] == 80)))
    out.append(("extrapolate_40C", by(lambda c: c["T"] >= 50), by(lambda c: c["T"] == 40)))
    for T in (50, 60, 70):
        out.append((f"interpolate_{T}C", by(lambda c, T=T: c["T"] != T), by(lambda c, T=T: c["T"] == T)))
    return out


def evaluate(pred, te):
    P = pred.reshape(len(te), len(T_EVAL)); r, v = [], []
    for p, c in zip(P, te):
        r.append(math.sqrt(np.mean((p - c["mr"]) ** 2)))
        v.append(bool(np.any(np.diff(p) > 0.01) or np.any(p < -0.02) or np.any(p > 1.05)))
    return float(np.mean(r)), 100.0 * float(np.mean(v))


def run_split(variant, spec, name, tr, te):
    """Fit the models of one variant on one split; returns {model key: (model name, rmse, inadmissible %)}."""
    budget_name, seed, refit = spec
    b = BUDGETS[budget_name]
    mech = Fick().fit(tr)
    keys = ("M1", "M1a", "M2", "M3", "M4", "M5", "M6", "M7") if refit is None else refit
    out = {}
    for key in keys:
        if key == "M1":
            m = mech
        elif key == "M1a":
            m = FickBook().fit(tr)
        elif key == "M2":
            m = Poly().fit(tr)
        elif key == "M3":
            m = MLP(b).fit(tr, seed=seed)
        elif key == "M4":
            m = GPR(b).fit(tr)
        elif key == "M5":
            m = Hybrid(b).fit(tr, mech)
        elif key == "M6":
            m = UDE(b).fit(tr, mech, seed=seed)
        else:
            m = LSTM(b).fit(tr, seed=seed)
        rmse, viol = evaluate(m.predict(te), te)
        out[key] = (m.name, rmse, viol)
    return variant, name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--budget-variants", action="store_true")
    a = ap.parse_args()
    if a.budget_variants:
        VARIANTS.update(BUDGET_VARIANTS)
    curves = load_curves()
    with open(HERE / "Ch04_Physical_Drying_Digitisation_Check.csv", "w", newline="", encoding="utf-8") as fh:
        rows = digitisation_check(curves)
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("digitisation: max |difference| =", max(abs(r["relative_difference_pct"]) for r in rows), "%")
    sp = splits(curves)
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_split, v, VARIANTS[v], name, tr, te) for v in VARIANTS for name, tr, te in sp]
        for fu in futs:
            v, name, out = fu.result(); res[(v, name)] = out
            print("finished", v, name, flush=True)
    folds = []
    for v, (budget_name, seed, refit) in VARIANTS.items():
        for name, tr, te in sp:
            merged = dict(res[("nominal_seed0", name)]); merged.update(res[(v, name)])
            for key in ("M1", "M1a", "M2", "M3", "M4", "M5", "M6", "M7"):
                mname, rmse, viol = merged[key]
                folds.append({"split": name, "n_train_curves": len(tr), "n_test_curves": len(te), "model": mname,
                              "rmse_moisture_ratio": round(rmse, 4), "inadmissible_curves_pct": round(viol, 1),
                              "variant": v, "training_budget": budget_name, "training_seed": seed,
                              "refitted_in_variant": "yes" if (refit is None or key in refit) else "copied_from_nominal_seed0"})
    with open(HERE / "Ch04_Physical_Drying_Folds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(folds[0].keys())); w.writeheader(); w.writerows(folds)
    summary = []
    groups = {"replicate_transfer": ("replicate_I_to_II", "replicate_II_to_I"), "temperature_extrapolation": ("extrapolate_80C", "extrapolate_40C"),
              "temperature_interpolation": ("interpolate_50C", "interpolate_60C", "interpolate_70C")}
    for v in VARIANTS:
        for gname, members in groups.items():
            block = []
            for model in sorted({f["model"] for f in folds}):
                fs = [f for f in folds if f["variant"] == v and f["split"] in members and f["model"] == model]
                block.append({"split_group": gname, "model": model, "rmse_mean": round(float(np.mean([f["rmse_moisture_ratio"] for f in fs])), 4),
                              "rmse_max_fold": max(f["rmse_moisture_ratio"] for f in fs),
                              "inadmissible_curves_mean_pct": round(float(np.mean([f["inadmissible_curves_pct"] for f in fs])), 1),
                              "n_folds": len(fs), "evidence_status": "published_physical_curves_one_laboratory_one_product_not_industrial_validation",
                              "variant": v, "training_budget": VARIANTS[v][0], "training_seed": VARIANTS[v][1]})
            for b_, rk in zip(block, rankdata([b_["rmse_mean"] for b_ in block], method="average")):
                b_["rank_in_variant"] = float(rk)
            summary += block
    with open(HERE / "Ch04_Physical_Drying_Summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    for s_ in summary:
        print(s_["variant"], s_["split_group"], s_["model"], s_["rmse_mean"], s_["inadmissible_curves_mean_pct"], "rank", s_["rank_in_variant"])


if __name__ == "__main__":
    main()
