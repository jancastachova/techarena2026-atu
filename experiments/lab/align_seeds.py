# Run from the repository root: .venv/bin/python experiments/lab/align_seeds.py
# Average the vector SEEDS instead of their scores. Each MF seed learns the same space up to an arbitrary rotation, so the
# 10 seeds of 22 / 26 are aligned to seed 0 by orthogonal Procrustes (impression-weighted), averaged (iterated: re-align to
# the running mean), then projected (top 8) and scored by cosine. Score averaging saturates at ~10 seeds (0.82383 projected);
# averaging the vectors removes noise BEFORE the nonlinear cosine. Day 6, MF of days 0-5.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
files = [ROOT / "data/cache/22_vectors_c7.npz"] + [ROOT / f"data/cache/26_vectors_c7_s{s}.npz" for s in range(1, 10)]
evd = np.flatnonzero(fit["day"] == 6); items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
H, n = e13.long_history(fit, fit, 100); H, n = H[evd], n[evd].astype(int)
m = (np.arange(100)[None] < n[:, None])[:, :, None]
unit = lambda X: X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
w = np.sqrt(np.bincount(fit["item"][np.repeat(fit["day"] < 6, 10)], minlength=n_items + 1)[:n_items].astype(np.float64))[:, None]
Ws = [np.load(f)["W_6"].astype(np.float64) for f in files]
def proj(W, k=8):
    _, U = np.linalg.eigh(W[1:].T @ W[1:]); U = U[:, ::-1][:, :k]; return (W @ U) @ U.T
def score(W):
    Wn = unit(W); Wn[0] = 0
    hv = unit((W[H] * m).sum(1))
    return np.einsum("ed,ekd->ek", hv, Wn[items]).ravel()
def procrustes(A, B):                       # rotation R minimising |w (A R - B)|
    u, _, vt = np.linalg.svd((w * A).T @ (w * B)); return u @ vt
mean = Ws[0].copy()
for it_ in range(3):
    aligned = [W @ procrustes(W, mean) for W in Ws]
    mean = np.mean(aligned, 0)
fit_q = np.mean([np.linalg.norm(w * (a - mean)) / np.linalg.norm(w * a) for a in aligned])
print(f"relative residual after alignment {fit_q:.3f}", flush=True)
for k in (1, 3, 5, 10):
    vm = np.mean(aligned[:k], 0)
    sa = np.mean([score(proj(W)) for W in Ws[:k]], 0)
    print(f"{k:2d} seeds: score-average (proj8) {auc(sa, y):.5f}   aligned vector average (proj8) {auc(score(proj(vm)), y):.5f}   "
          f"aligned average, no projection {auc(score(vm), y):.5f}", flush=True)
