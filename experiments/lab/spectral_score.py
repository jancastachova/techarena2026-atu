# Run from the repository root: .venv/bin/python experiments/lab/spectral_score.py [--in data/cache/47_spec_c7.npz]
# LEARNED SPECTRAL SCORE (2026-11-03): one column for the trees. LightGBM did not use the 16 spectral columns (day 4 -0.00037)
# but gained from single good scores (the projected cosine). Here the per-direction weights of the spectral view
# (lab/spectral_view.py) are fitted by a logistic regression on the rows of EARLIER days only (time-forward): day d's score
# uses weights fitted on days 2..d-1 (day 2: the projected cosine, top 8 equal weights); the test rows use days 2-6.
# Inputs per row: f_k and f_k minus its event mean (within-event contrast), k < K. Output: 4-column extra-group cache
# [score, 0, 0, score] (24_ensemble --extra / lab/extra_member.py --base-extra).
import argparse, importlib.util
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT = Path.cwd()
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="data/cache/47_spec_c7.npz")
ap.add_argument("--out", default="data/cache/47_specw_c7.npz"); ap.add_argument("--l2", type=float, default=1.0)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))[0]
Z = np.load(ROOT / a.inp); F, Ft = Z["mf_fit"], Z["mf_test"]
y = fit["label"].astype(np.float64); day = np.repeat(fit["day"], 10)


def design(M):
    E = M.reshape(-1, 10, M.shape[1])
    return np.concatenate([M, (E - E.mean(1, keepdims=True)).reshape(M.shape)], 1).astype(np.float64)


def fit_lr(X, t):
    rng = np.random.default_rng(0); s = rng.choice(len(t), min(1_500_000, len(t)), replace=False); X, t = X[s], t[s]
    X1 = np.c_[X, np.ones(len(X))]
    def f(w):
        z = X1 @ w; p = 1 / (1 + np.exp(-z))
        return (np.logaddexp(0, z) - t * z).mean() + a.l2 * 1e-4 * (w[:-1] ** 2).sum(), X1.T @ (p - t) / len(t) + np.r_[2 * a.l2 * 1e-4 * w[:-1], 0]
    return minimize(f, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B").x


out = np.full((len(F), 4), np.nan, np.float32)
K = F.shape[1]
for d in range(2, 7):
    r = np.flatnonzero(day == d)
    if d == 2:
        s = F[r, :8].sum(1)
    else:
        tr = np.flatnonzero((day >= 2) & (day < d))
        w = fit_lr(design(F[tr]), y[tr])
        s = np.c_[design(F[r]), np.ones(len(r))] @ w
        e13.log(f"day {d}: weights (first {K}) " + " ".join(f"{x:.2f}" for x in w[:K]))
    out[r, 0] = out[r, 3] = s; out[r, 1] = out[r, 2] = 0
    e13.log(f"day {d}: AUC learned {e13.roc_auc(s, y[r]):.5f}  top8 {e13.roc_auc(F[r, :8].sum(1), y[r]):.5f}")
tr = np.flatnonzero(day >= 2)
w = fit_lr(design(F[tr]), y[tr]); st = np.c_[design(Ft), np.ones(len(Ft))] @ w
t = np.zeros((len(Ft), 4), np.float32); t[:, 0] = t[:, 3] = st
np.savez(ROOT / a.out, mf_fit=out, mf_test=t)
e13.log(f"wrote {a.out}")
