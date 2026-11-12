"""Executed in-silico comparison of mechanistic, statistical and hybrid architectures (Chapter 4), with a
one-factor-at-a-time sensitivity study of the rankings.

Task: predict the mean dry-basis moisture X(t) of a thin apple slab during convective drying from the
information that is permitted at inference: time, air temperature, air relative humidity and the measured
initial moisture. The reference simulator (the "generator") is a one-dimensional Fickian slab with a
moisture-dependent diffusivity and GAB surface equilibrium; it is never available to a predictor.

Architectures actually trained or calibrated here
  M1 Mechanistic     constant-diffusivity Fickian series, two parameters (ln D0, Ea) calibrated by least squares
  M2 Polynomial      ridge polynomial in (t, T, RH, X0); degree and penalty chosen by batch-grouped CV
  M3 MLP             scikit-learn MLPRegressor; architecture and penalty chosen by batch-grouped CV, restarts averaged
  M4 GPR             Gaussian-process regression with ARD Matern kernel; marginal-likelihood hyperparameters
  M5 Parallel hybrid mechanistic M1 + GPR on its residual
  M6 UDE             lumped first-order mechanistic term with a neural multiplicative correction, trained
                     through the ODE (least squares on whole trajectories)
  M7 LSTM            one-layer LSTM (PyTorch, CPU) over the observation sequence of a batch; weight decay chosen by
                     batch-grouped CV; mean over training seeds

Protocol: whole batches are withheld (batch = one drying trajectory); hyperparameters are selected only by grouped
cross-validation inside the training batches; each independent repeat draws new batches from the Monte Carlo
disturbance table. Test sets: (ID) 40 withheld batches in the training condition range; (OOD) 40 batches at higher
air temperature and humidity than any training batch; (CLOSURE_SHIFT) 40 batches at training conditions from a
generator whose diffusivity closure differs from the training generator.

Scenarios (one factor changed at a time; "baseline" reproduces Table 4.6):
  baseline        40 training batches, noise sd 0.03 kg/kg, nominal training budget, closure A, surface at equilibrium
  budget_low      MLP at most 10 epochs and 1 restart; LSTM 100 epochs and 1 seed; UDE 15 function evaluations;
                  GPR without optimiser restarts
  budget_high     MLP up to 3000 epochs with tolerance 1e-6 over 50 epochs and 5 restarts; LSTM 1600 epochs and 5 seeds;
                  UDE 180 function evaluations; GPR with 4 optimiser restarts
  ntrain_10/20/80 10, 20 or 80 training batches
  noise_0.01/0.06 measurement-noise standard deviation 0.01 or 0.06 kg/kg
  gen_closure_B   training, ID and OOD batches from closure B (power law); the shift set uses closure A
  gen_surface_resistance  closure A with a finite surface mass-transfer coefficient (1e-7 m/s, an assumed value giving a
                  mass Biot number of about 0.9 at 65 degC); the shift set uses closure B with the same surface condition
The nominal MLP stops after about 20-35 epochs through the default scikit-learn convergence test (tolerance 1e-4 over
10 epochs); max_iter = 3000 is not reached. The budget scenarios vary this stopping rule.

Repeats: 10 per scenario (--repeats N changes this for every scenario).

Scope: results hold for these generators only. They establish neither physical accuracy nor performance on
real drying data.

Outputs (next to this script): Ch04_Executed_Drying_Comparison_Repeats.csv and
Ch04_Executed_Drying_Comparison_Summary.csv (one row per scenario, condition and model, with mean RMSE, a bootstrap
95% interval of the mean over repeats, ranks within repeats and Kendall's tau against the baseline order).
Run: python benchmark_ch04_drying_architectures.py [--scenarios baseline,budget_low,...] [--workers N] [--cache file]
"""
import argparse
import csv
import json
import math
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import torch  # must be imported before numpy/scipy/sklearn on Windows (DLL load order)
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.linalg import solve_banded
from scipy.optimize import least_squares
from scipy.stats import kendalltau, rankdata, wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

R_GAS = 8.314
L_HALF = 0.0015                # slab half-thickness (m)
D0_PRE, EA_D = 4.2e-6, 28400.0  # m2/s, J/mol  (PD-080 D_eff0, PD-081 Ea of data/parameter_provenance_register.csv)
XM, C_GAB, K_GAB = 0.082, 11.4, 0.84   # GAB constants Xm, C, K (PD-082..PD-084; PD-085 is SEC_base_drying, not used here)
T_GRID = np.arange(1.0, 16.01, 1.0)    # observation times (h)
SIGMA_OBS = 0.03               # measurement noise sd (kg/kg db), baseline
X_TARGET = 1.0                 # mid-drying moisture used for the drying-time decision (kg/kg db)
N_TRAIN, N_TEST, N_REPEATS = 40, 40, 10
K_SURF = 1.0e-7                # surface mass-transfer coefficient of the gen_surface_resistance scenario (m/s), assumed

BUDGETS = {
    "nominal": {"mlp_max_iter": 3000, "mlp_tol": 1e-4, "mlp_no_change": 10, "mlp_restarts": 3,
                "gpr_restarts": 1, "ude_nfev": 60, "lstm_epochs": 400, "lstm_seeds": 3},
    "low": {"mlp_max_iter": 10, "mlp_tol": 1e-4, "mlp_no_change": 10, "mlp_restarts": 1,
            "gpr_restarts": 0, "ude_nfev": 15, "lstm_epochs": 100, "lstm_seeds": 1},
    "high": {"mlp_max_iter": 3000, "mlp_tol": 1e-6, "mlp_no_change": 50, "mlp_restarts": 5,
             "gpr_restarts": 4, "ude_nfev": 180, "lstm_epochs": 1600, "lstm_seeds": 5},
}
BASE = {"factor": "none", "n_train": N_TRAIN, "sigma": SIGMA_OBS, "budget": "nominal",
        "train_closure": "A", "shift_closure": "B", "surface": "equilibrium"}
SCENARIOS = {
    "baseline": {},
    "budget_low": {"factor": "training_budget", "budget": "low"},
    "budget_high": {"factor": "training_budget", "budget": "high"},
    "ntrain_10": {"factor": "sample_size", "n_train": 10},
    "ntrain_20": {"factor": "sample_size", "n_train": 20},
    "ntrain_80": {"factor": "sample_size", "n_train": 80},
    "noise_0.01": {"factor": "noise", "sigma": 0.01},
    "noise_0.06": {"factor": "noise", "sigma": 0.06},
    "gen_closure_B": {"factor": "generator", "train_closure": "B", "shift_closure": "A"},
    "gen_surface_resistance": {"factor": "generator", "surface": "resistance"},
}
COST_ORDER = ["budget_high", "ntrain_80", "baseline", "noise_0.01", "noise_0.06", "gen_closure_B",
              "gen_surface_resistance", "ntrain_20", "ntrain_10", "budget_low"]


def scenario_cfg(name):
    cfg = dict(BASE); cfg.update(SCENARIOS[name]); cfg["name"] = name
    return cfg


# ----------------------------------------------------------------------------- generator
def x_eq(rh_pct):
    aw = np.clip(np.asarray(rh_pct, float) / 100.0, 1e-4, 0.95)
    return XM * C_GAB * K_GAB * aw / ((1 - K_GAB * aw) * (1 - K_GAB * aw + C_GAB * K_GAB * aw))


def simulate(T_c, rh, x0, dmult, closure="A", n_cell=20, dt_s=120.0, surface="equilibrium"):
    """Mean moisture at T_GRID for one batch (finite volume, semi-implicit). Surface: Dirichlet at X_eq, or a
    finite mass-transfer coefficient K_SURF in series with the half-cell diffusion resistance."""
    T = T_c + 273.15
    d_t = D0_PRE * math.exp(-EA_D / (R_GAS * T)) * dmult
    xe = float(x_eq(rh))
    dx = L_HALF / n_cell
    X = np.full(n_cell, x0)
    out, t, k_out = [], 0.0, 0
    n_steps = int(round(T_GRID[-1] * 3600 / dt_s))
    for step in range(1, n_steps + 1):
        if closure == "A":
            f = np.exp(-1.2 * (1.0 - np.clip(X, 0, None) / x0))
        else:                                     # closure B: power-law moisture dependence
            f = (np.clip(X, 0, None) / x0 + 0.05) ** 0.8
        D = d_t * f
        Df = 0.5 * (D[1:] + D[:-1])
        r = dt_s / dx ** 2
        lower = np.zeros(n_cell); diag = np.ones(n_cell); upper = np.zeros(n_cell)
        diag[:-1] += r * Df; upper[:-1] = -r * Df
        diag[1:] += r * Df; lower[1:] = -r * Df
        b = X.copy()
        Ds = D[-1]
        if surface == "equilibrium":              # surface cell: Dirichlet at distance dx/2
            diag[-1] += 2 * r * Ds
            b[-1] += 2 * r * Ds * xe
        else:                                     # conductance of half cell and external film in series
            g = 1.0 / (dx / (2 * Ds) + 1.0 / K_SURF)
            diag[-1] += dt_s / dx * g
            b[-1] += dt_s / dx * g * xe
        ab = np.vstack([np.r_[0.0, upper[:-1]], diag, np.r_[lower[1:], 0.0]])
        X = solve_banded((1, 1), ab, b)
        t = step * dt_s / 3600.0
        if k_out < len(T_GRID) and abs(t - T_GRID[k_out]) < 1e-9:
            out.append(X.mean()); k_out += 1
    return np.array(out)


def make_batches(rows, ranges, closure, rng, sigma=SIGMA_OBS, surface="equilibrium"):
    batches = []
    for r in rows:
        T = rng.uniform(*ranges["T"]); rh = rng.uniform(*ranges["RH"])
        x0 = float(r["drying_X0_db"]); dm = float(r["drying_Deff_mult"])
        truth = simulate(T, rh, x0, dm, closure, surface=surface)
        obs = truth + rng.normal(0, sigma, truth.shape)
        batches.append({"T": T, "RH": rh, "X0": x0, "truth": truth, "obs": obs})
    return batches


def features(batches):
    rows = []
    for k, b in enumerate(batches):
        for i, t in enumerate(T_GRID):
            rows.append([math.log(t), b["T"], b["RH"], b["X0"]])
    return np.array(rows)


def _xe_x0(batches):
    xe = np.repeat([float(x_eq(b["RH"])) for b in batches], len(T_GRID))
    x0 = np.repeat([b["X0"] for b in batches], len(T_GRID))
    return xe, x0


def to_mr(batches, y):
    """Moisture-ratio scaling (X - Xeq)/(X0 - Xeq), the usual target of empirical drying models."""
    xe, x0 = _xe_x0(batches)
    return (y - xe) / (x0 - xe)


def from_mr(batches, r):
    xe, x0 = _xe_x0(batches)
    return xe + r * (x0 - xe)


def groups(batches):
    return np.repeat(np.arange(len(batches)), len(T_GRID))


def targets(batches):
    return np.concatenate([b["obs"] for b in batches])


# ----------------------------------------------------------------------------- mechanistic
def fick_series(t_h, T_c, rh, x0, ln_d0, ea):
    T = T_c + 273.15
    D = math.exp(ln_d0) * np.exp(-ea / (R_GAS * T))
    xe = float(x_eq(rh))
    n = np.arange(0, 60)[:, None]
    tau = (2 * n + 1) ** 2 * math.pi ** 2 * D * (np.asarray(t_h)[None, :] * 3600) / (4 * L_HALF ** 2)
    ser = (8 / math.pi ** 2) * np.sum(np.exp(-tau) / (2 * n + 1) ** 2, axis=0)
    return xe + (x0 - xe) * ser


class Mechanistic:
    name = "M1_Mechanistic_Fickian_constant_D"

    def fit(self, batches):
        def res(p):
            return np.concatenate([fick_series(T_GRID, b["T"], b["RH"], b["X0"], p[0], p[1]) - b["obs"] for b in batches])
        sol = least_squares(res, x0=[math.log(D0_PRE) - 1.0, EA_D], x_scale=[1.0, 1e4])
        self.p = sol.x
        return self

    def predict_batch(self, b):
        return fick_series(T_GRID, b["T"], b["RH"], b["X0"], *self.p)

    def predict(self, batches):
        return np.concatenate([self.predict_batch(b) for b in batches])


# ----------------------------------------------------------------------------- statistical
class Poly:
    name = "M2_Polynomial_ridge"

    def fit(self, batches):
        X, g = features(batches), groups(batches)
        y = to_mr(batches, targets(batches))
        best = None
        for deg in (2, 3, 4):
            for alpha in (1e-3, 1e-1, 10.0):
                err = []
                for tr, va in GroupKFold(4).split(X, y, g):
                    m = make_pipeline(StandardScaler(), PolynomialFeatures(deg), Ridge(alpha=alpha)).fit(X[tr], y[tr])
                    err.append(np.mean((m.predict(X[va]) - y[va]) ** 2))
                if best is None or np.mean(err) < best[0]:
                    best = (np.mean(err), deg, alpha)
        self.cfg = best[1:]
        self.m = make_pipeline(StandardScaler(), PolynomialFeatures(best[1]), Ridge(alpha=best[2])).fit(X, y)
        return self

    def predict(self, batches):
        return from_mr(batches, self.m.predict(features(batches)))


class MLP:
    name = "M3_MLP"

    def __init__(self, budget=None):
        self.b = budget or BUDGETS["nominal"]

    def _mk(self, hidden, alpha, seed):
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", alpha=alpha,
                                                            max_iter=self.b["mlp_max_iter"], tol=self.b["mlp_tol"],
                                                            n_iter_no_change=self.b["mlp_no_change"],
                                                            random_state=seed, early_stopping=False))

    def fit(self, batches, seed=0):
        X, g = features(batches), groups(batches)
        y = to_mr(batches, targets(batches))
        best = None
        for hidden in ((16,), (32, 32)):
            for alpha in (1e-4, 1e-2):
                err = []
                for tr, va in GroupKFold(4).split(X, y, g):
                    m = self._mk(hidden, alpha, seed).fit(X[tr], y[tr])
                    err.append(np.mean((m.predict(X[va]) - y[va]) ** 2))
                if best is None or np.mean(err) < best[0]:
                    best = (np.mean(err), hidden, alpha)
        self.cfg = best[1:]
        self.models = [self._mk(best[1], best[2], seed + s).fit(X, y) for s in range(self.b["mlp_restarts"])]
        self.epochs = float(np.mean([m.steps[-1][1].n_iter_ for m in self.models]))
        return self

    def predict(self, batches):
        X = features(batches)
        return from_mr(batches, np.mean([m.predict(X) for m in self.models], axis=0))


def gpr_model(n_restarts=1):
    k = ConstantKernel(0.5) * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=n_restarts, random_state=0))


class GPR:
    name = "M4_GPR"

    def __init__(self, budget=None):
        self.b = budget or BUDGETS["nominal"]

    def fit(self, batches):
        self.m = gpr_model(self.b["gpr_restarts"]).fit(features(batches), to_mr(batches, targets(batches)))
        return self

    def predict(self, batches, return_std=False):
        X = self.m[:-1].transform(features(batches))
        mu, sd = self.m[-1].predict(X, return_std=True)
        xe, x0 = _xe_x0(batches)
        mu, sd = from_mr(batches, mu), sd * (x0 - xe)
        return (mu, sd) if return_std else mu


class ParallelHybrid:
    name = "M5_Parallel_hybrid_M1_plus_GPR_residual"

    def __init__(self, budget=None):
        self.b = budget or BUDGETS["nominal"]

    def fit(self, batches, mech):
        self.mech = mech
        X = features(batches)
        resid = targets(batches) - mech.predict(batches)
        Xa = np.c_[X, mech.predict(batches)]
        k = ConstantKernel(0.1) * Matern(length_scale=np.ones(5), length_scale_bounds=(1e-2, 1e3), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
        self.m = make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=self.b["gpr_restarts"],
                                                                          random_state=0)).fit(Xa, resid)
        return self

    def predict(self, batches):
        base = self.mech.predict(batches)
        return base + self.m.predict(np.c_[features(batches), base])


# ----------------------------------------------------------------------------- UDE
NH = 6


def _unpack(p):
    lnk0, ea = p[0], p[1]
    w1 = p[2:2 + 3 * NH].reshape(NH, 3); b1 = p[2 + 3 * NH:2 + 4 * NH]
    w2 = p[2 + 4 * NH:2 + 5 * NH]; b2 = p[2 + 5 * NH]
    return lnk0, ea, w1, b1, w2, b2


def ude_rollout(p, T_c, rh, x0):
    """Vectorised RK4 over batches; returns X at T_GRID (shape n_batch x n_t)."""
    lnk0, ea, w1, b1, w2, b2 = _unpack(p)
    T = np.asarray(T_c) + 273.15; xe = x_eq(rh); x0 = np.asarray(x0)
    k = np.exp(lnk0) * np.exp(-ea / (R_GAS * T))                      # 1/h
    ts, rs = (np.asarray(T_c) - 65.0) / 10.0, (np.asarray(rh) - 35.0) / 20.0

    def f(X):
        z = np.stack([X / x0, ts, rs], axis=1)
        g = np.tanh(z @ w1.T + b1) @ w2 + b2
        return -k * (1.0 + g) * (X - xe)

    dt = 0.1; X = x0.copy(); out = []
    for step in range(1, int(round(T_GRID[-1] / dt)) + 1):
        k1 = f(X); k2 = f(X + 0.5 * dt * k1); k3 = f(X + 0.5 * dt * k2); k4 = f(X + dt * k3)
        X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if len(out) < len(T_GRID) and abs(step * dt - T_GRID[len(out)]) < 1e-9:
            out.append(X.copy())
    return np.stack(out, axis=1)


class UDE:
    name = "M6_UDE_neural_correction"

    def __init__(self, budget=None):
        self.b = budget or BUDGETS["nominal"]

    def _fit_one(self, batches, lam, mech, rng):
        Tc = np.array([b["T"] for b in batches]); rh = np.array([b["RH"] for b in batches]); x0 = np.array([b["X0"] for b in batches])
        obs = np.stack([b["obs"] for b in batches])
        # lumped rate constant matched to the leading Fourier mode of the calibrated mechanistic model
        lnk0 = mech.p[0] + math.log(math.pi ** 2 / (4 * L_HALF ** 2) * 3600.0)
        p0 = np.r_[lnk0, mech.p[1], rng.normal(0, 0.1, 3 * NH), np.zeros(NH), rng.normal(0, 0.1, NH), 0.0]

        def res(p):
            r = (ude_rollout(p, Tc, rh, x0) - obs).ravel()
            return np.r_[r, math.sqrt(lam) * p[2:]]
        sol = least_squares(res, p0, x_scale=np.r_[1.0, 1e4, np.ones(len(p0) - 2)], max_nfev=self.b["ude_nfev"])
        return sol.x

    def fit(self, batches, mech, seed=0):
        rng = np.random.default_rng(seed)
        best = None
        idx = np.arange(len(batches))
        for lam in (1e-3, 1e-1):
            err = []
            for tr, va in GroupKFold(4).split(idx, groups=idx):
                p = self._fit_one([batches[i] for i in tr], lam, mech, rng)
                v = [batches[i] for i in va]
                pred = ude_rollout(p, [b["T"] for b in v], [b["RH"] for b in v], [b["X0"] for b in v])
                err.append(np.mean((pred - np.stack([b["obs"] for b in v])) ** 2))
            if best is None or np.mean(err) < best[0]:
                best = (np.mean(err), lam)
        self.lam = best[1]
        self.p = self._fit_one(batches, self.lam, mech, rng)
        return self

    def predict(self, batches):
        out = ude_rollout(self.p, [b["T"] for b in batches], [b["RH"] for b in batches], [b["X0"] for b in batches])
        return out.ravel()


# ----------------------------------------------------------------------------- LSTM
class LSTM:
    """One-layer LSTM over the observation sequence of a batch; per-step inputs (log t, T, RH, X0), output = moisture ratio.
    Weight decay is chosen by batch-grouped cross-validation; the prediction is the mean over training seeds."""
    name = "M7_LSTM"

    def __init__(self, n_in=4, hidden=16, epochs=400, lr=0.02, n_seeds=3):
        self.n_in, self.hidden, self.epochs, self.lr, self.n_seeds = n_in, hidden, epochs, lr, n_seeds

    def _seq(self, batches):
        return features(batches).reshape(len(batches), len(T_GRID), -1)

    def _train(self, Xs, Ys, wd, seed):
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(1)

        class Net(torch.nn.Module):
            def __init__(s, n_in, h):
                super().__init__()
                s.l = torch.nn.LSTM(n_in, h, batch_first=True)
                s.o = torch.nn.Linear(h, 1)

            def forward(s, x):
                return s.o(s.l(x)[0]).squeeze(-1)

        net = Net(self.n_in, self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=wd)
        X = torch.tensor(Xs, dtype=torch.float32); Y = torch.tensor(Ys, dtype=torch.float32)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = torch.mean((net(X) - Y) ** 2)
            loss.backward()
            opt.step()
        return net

    def _predict_net(self, net, Xs):
        import torch
        with torch.no_grad():
            return net(torch.tensor(Xs, dtype=torch.float32)).numpy()

    def fit(self, batches, seed=0):
        Xs = self._seq(batches)
        self.mu, self.sd = Xs.reshape(-1, Xs.shape[-1]).mean(0), Xs.reshape(-1, Xs.shape[-1]).std(0) + 1e-9
        Xn = (Xs - self.mu) / self.sd
        Y = to_mr(batches, targets(batches)).reshape(len(batches), len(T_GRID))
        idx = np.arange(len(batches)); best = None
        for wd in (1e-4, 1e-2):
            err = []
            for tr, va in GroupKFold(4).split(idx, groups=idx):
                net = self._train(Xn[tr], Y[tr], wd, seed)
                err.append(np.mean((self._predict_net(net, Xn[va]) - Y[va]) ** 2))
            if best is None or np.mean(err) < best[0]:
                best = (np.mean(err), wd)
        self.wd = best[1]
        self.nets = [self._train(Xn, Y, self.wd, seed + s) for s in range(self.n_seeds)]
        return self

    def predict(self, batches):
        Xn = (self._seq(batches) - self.mu) / self.sd
        P = np.mean([self._predict_net(n, Xn) for n in self.nets], axis=0)
        return from_mr(batches, P.ravel())


# ----------------------------------------------------------------------------- metrics
def per_batch(pred, batches):
    P = pred.reshape(len(batches), len(T_GRID)); out = {"rmse": [], "viol": [], "dt_err": []}
    for p, b in zip(P, batches):
        out["rmse"].append(math.sqrt(np.mean((p - b["truth"]) ** 2)))
        xe = float(x_eq(b["RH"]))
        out["viol"].append(bool(np.any(np.diff(p) > 1e-3) or np.any(p < xe - 0.02) or np.any(p < 0)))
        out["dt_err"].append(abs(_t_cross(p) - _t_cross(b["truth"])))
    return out


def _t_cross(curve):
    below = np.where(curve <= X_TARGET)[0]
    if len(below) == 0:
        return T_GRID[-1]
    i = below[0]
    if i == 0:
        return T_GRID[0]
    return T_GRID[i - 1] + (curve[i - 1] - X_TARGET) / (curve[i - 1] - curve[i]) * (T_GRID[i] - T_GRID[i - 1])


def run_repeat(rep, rows, scen="baseline"):
    cfg = scenario_cfg(scen); bud = BUDGETS[cfg["budget"]]
    n_tr, sig, srf = cfg["n_train"], cfg["sigma"], cfg["surface"]
    rng = np.random.default_rng(1000 + rep)
    pick = rng.permutation(len(rows))
    sel = lambda a, b: [rows[i] for i in pick[a:b]]
    id_rng = {"T": (55, 70), "RH": (15, 50)}
    train = make_batches(sel(0, n_tr), id_rng, cfg["train_closure"], rng, sig, srf)
    tests = {"ID": make_batches(sel(n_tr, n_tr + N_TEST), id_rng, cfg["train_closure"], rng, sig, srf),
             "OOD": make_batches(sel(n_tr + N_TEST, n_tr + 2 * N_TEST), {"T": (72, 80), "RH": (55, 75)}, cfg["train_closure"], rng, sig, srf),
             "CLOSURE_SHIFT": make_batches(sel(n_tr + 2 * N_TEST, n_tr + 3 * N_TEST), id_rng, cfg["shift_closure"], rng, sig, srf)}
    mech = Mechanistic().fit(train)
    models = {"M1": mech, "M2": Poly().fit(train), "M3": MLP(bud).fit(train, seed=rep), "M4": GPR(bud).fit(train)}
    models["M5"] = ParallelHybrid(bud).fit(train, mech)
    models["M6"] = UDE(bud).fit(train, mech, seed=rep)
    models["M7"] = LSTM(epochs=bud["lstm_epochs"], n_seeds=bud["lstm_seeds"]).fit(train, seed=rep)
    rows_out = []
    for cond, tb in tests.items():
        for key, m in models.items():
            pred = m.predict(tb)
            pb = per_batch(pred, tb)
            row = {"scenario": scen, "factor": cfg["factor"], "n_train_batches": n_tr, "noise_sd_kg_kg": sig,
                   "training_budget": cfg["budget"], "train_closure": cfg["train_closure"],
                   "shift_closure": cfg["shift_closure"], "surface_condition": srf,
                   "repeat": rep, "condition": cond, "model": m.name, "rmse_kg_kg_db": float(np.mean(pb["rmse"])),
                   "violation_rate_pct": 100.0 * float(np.mean(pb["viol"])), "drying_time_abs_error_h": float(np.mean(pb["dt_err"])),
                   "gpr_90pct_coverage": "", "mlp_epochs_mean": round(models["M3"].epochs, 1) if key == "M3" else ""}
            if key == "M4":
                mu, sd = m.predict(tb, return_std=True)
                truth = np.concatenate([b["truth"] for b in tb])
                row["gpr_90pct_coverage"] = float(np.mean(np.abs(truth - mu) <= 1.645 * sd))
            rows_out.append(row)
    return rows_out


def _boot_ci(x, n_boot=2000, seed=12345):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(allrows):
    # rank of each model within each scenario x condition x repeat (1 = lowest RMSE)
    keyed = {}
    for r in allrows:
        keyed.setdefault((r["scenario"], r["condition"], r["repeat"]), []).append(r)
    for rs in keyed.values():
        for r, rk in zip(rs, rankdata([r["rmse_kg_kg_db"] for r in rs], method="average")):
            r["rank_in_repeat"] = float(rk)
    models = sorted({r["model"] for r in allrows})
    scen_order = [s for s in SCENARIOS if any(r["scenario"] == s for r in allrows)]
    summary = []
    base_mean = {}
    for scen in scen_order:
        for cond in ("ID", "OOD", "CLOSURE_SHIFT"):
            srows = [r for r in allrows if r["scenario"] == scen and r["condition"] == cond]
            base = {r["repeat"]: r["rmse_kg_kg_db"] for r in srows if r["model"].startswith("M1")}
            means = {}
            block = []
            for name in models:
                rs = sorted([r for r in srows if r["model"] == name], key=lambda r: r["repeat"])
                rm = np.array([r["rmse_kg_kg_db"] for r in rs]); vi = np.array([r["violation_rate_pct"] for r in rs])
                dt = np.array([r["drying_time_abs_error_h"] for r in rs]); rk = np.array([r["rank_in_repeat"] for r in rs])
                diff = np.array([r["rmse_kg_kg_db"] - base[r["repeat"]] for r in rs])
                p = ""
                if not name.startswith("M1") and np.any(diff != 0):
                    p = round(float(wilcoxon(diff).pvalue), 4)
                cov = [r["gpr_90pct_coverage"] for r in rs if r["gpr_90pct_coverage"] != ""]
                lo, hi = _boot_ci(rm)
                means[name] = rm.mean()
                cfg = scenario_cfg(scen)
                block.append({"scenario": scen, "factor": cfg["factor"], "n_train_batches": cfg["n_train"], "noise_sd_kg_kg": cfg["sigma"],
                              "training_budget": cfg["budget"], "train_closure": cfg["train_closure"], "shift_closure": cfg["shift_closure"],
                              "surface_condition": cfg["surface"], "condition": cond, "model": name,
                              "rmse_mean": round(rm.mean(), 4), "rmse_sd_over_repeats": round(rm.std(ddof=1), 4),
                              "rmse_min": round(rm.min(), 4), "rmse_max": round(rm.max(), 4),
                              "rmse_mean_ci95_low": round(lo, 4), "rmse_mean_ci95_high": round(hi, 4),
                              "rank_mean": round(rk.mean(), 2), "rank_min": rk.min(), "rank_max": rk.max(),
                              "best_in_repeats_pct": round(100.0 * float(np.mean(rk == 1.0)), 1),
                              "violation_rate_mean_pct": round(vi.mean(), 1), "drying_time_abs_error_mean_h": round(dt.mean(), 3),
                              "paired_diff_vs_M1_mean": round(diff.mean(), 4), "wilcoxon_p_vs_M1": p,
                              "gpr_90pct_coverage_mean": round(float(np.mean(cov)), 3) if cov else "",
                              "kendall_tau_vs_baseline": "", "n_repeats": len(rs),
                              "evidence_status": "executed_in_silico_comparison_on_declared_generators_not_physical_validation"})
            if scen == "baseline":
                base_mean[cond] = means
            if cond in base_mean:
                tau = kendalltau([base_mean[cond][m] for m in models], [means[m] for m in models]).statistic
                for b in block:
                    b["kendall_tau_vs_baseline"] = round(float(tau), 3)
            summary += block
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--cache", default="", help="optional JSON-lines file of finished jobs (resume)")
    a = ap.parse_args()
    names = list(SCENARIOS) if a.scenarios == "all" else a.scenarios.split(",")
    with open(DATA / "monte_carlo_disturbances_seed42.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    done = {}
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, encoding="utf-8") as fh:
            for line in fh:
                j = json.loads(line); done[(j["scenario"], j["repeat"])] = j["rows"]
    nrep = {s: a.repeats for s in names}
    jobs = sorted([(s, r) for s in names for r in range(nrep[s]) if (s, r) not in done], key=lambda j: (j[1], COST_ORDER.index(j[0])))
    t = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
            futs = {ex.submit(run_repeat, r, rows, s): (s, r) for s, r in jobs}
            for fu in as_completed(futs):
                s, r = futs[fu]
                res = fu.result(); done[(s, r)] = res
                if a.cache:
                    with open(a.cache, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"scenario": s, "repeat": r, "rows": res}) + "\n")
                print("finished", s, r, "after", round(time.time() - t), "s", flush=True)
    allrows = [row for s in names for r in range(nrep[s]) for row in done[(s, r)]]
    summary = summarise(allrows)
    with open(HERE / "Ch04_Executed_Drying_Comparison_Repeats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    with open(HERE / "Ch04_Executed_Drying_Comparison_Summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    for s in summary:
        print(s["scenario"], s["condition"], s["model"][:2], s["rmse_mean"], "+-", s["rmse_sd_over_repeats"],
              "rank", s["rank_mean"], f"[{s['rank_min']}-{s['rank_max']}]", "viol", s["violation_rate_mean_pct"],
              "tau", s["kendall_tau_vs_baseline"])


if __name__ == "__main__":
    main()
