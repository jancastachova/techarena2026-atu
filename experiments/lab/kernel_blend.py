# Run from the repository root: .venv/bin/python experiments/lab/kernel_blend.py <kernel_aud --out npz> [--day 6]
# Does the kernel audience add to the two cosines? Logistic regression on day-D half A (rng(0) event halves), AUC on half B:
# own cos, aud cos, log-odds item click rate, each also relative to its event mean; then + the kernel scores.
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--day", type=int, default=6); a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
Z = np.load(a.npz)
evd = Z["evd"]; y = fit["label"].reshape(-1, 10)[evd].ravel().astype(float)
rng = np.random.default_rng(0); half = rng.random(len(evd)) < 0.5; A = np.repeat(half, 10); B = ~A
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def ev_rel(v):
    r = v.reshape(-1, 10); return (r - r.mean(1, keepdims=True)).ravel()
base = [Z["own_cos"], Z["aud_cos"], lg(Z["item_ctr"]), ev_rel(Z["own_cos"]), ev_rel(Z["aud_cos"]), ev_rel(lg(Z["item_ctr"]))]
def fitlr(F, y, it=30):
    X = np.c_[np.ones(len(y)), F]; w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (y - p); Hm = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        w += np.linalg.solve(Hm, g)
    return w
for nm, extra in [("base", []), ("+k2", ["kernel_k2"]), ("+k5", ["kernel_k5"]), ("+k10", ["kernel_k10"]),
                  ("+k2 k5 k10", ["kernel_k2", "kernel_k5", "kernel_k10"])]:
    F = np.c_[tuple(base + [lg(Z[e]) for e in extra] + [ev_rel(lg(Z[e])) for e in extra])]
    mu, sd = F[A].mean(0), F[A].std(0) + 1e-9; F = (F - mu) / sd
    w = fitlr(F[A], y[A]); s = np.c_[np.ones(B.sum()), F[B]] @ w
    print(f"  logistic on half A, half-B AUC {nm:12s} {e13.roc_auc(s, y[B]):.5f}")
