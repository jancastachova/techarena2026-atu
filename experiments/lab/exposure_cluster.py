# Run from the repository root: .venv/bin/python experiments/lab/exposure_cluster.py
# 2026-11-04: are the items in EXPOSURE space discrete categories? Uses the SVD saved by lab/exposure_cat.py (all fit days).
# Well-shown items only. (1) k-means on unit 8-dim embeddings vs a Gaussian of the same covariance. (2) histogram of the top
# cosine. (3) canonical correlations between exposure space and the MF's top-8 item space. (4) k-means labels vs MF.
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
Z = np.load(sys.argv[1]); S, V, di = Z["S"], Z["V"].astype(np.float64), Z["di"]
rng = np.random.default_rng(0)
good = np.flatnonzero(di >= 100); print(f"items with >= 100 impressions: {len(good):,} of {(di > 0).sum():,}")
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
def kmeans(A, k, it=40, starts=4):
    best, bl = -1, None
    for s in range(starts):
        C = A[rng.choice(len(A), k, replace=False)]
        for _ in range(it):
            l = np.argmax(A @ C.T, 1)
            C = unit(np.stack([A[l == j].mean(0) if (l == j).any() else C[j] for j in range(k)]))
        v = np.max(A @ C.T, 1).mean()
        if v > best: best, bl, bc = v, l, C
    return best, bl, bc
for r in (7, 8, 9):
    E = V[good, 1:r + 1] / np.sqrt(di[good])[:, None]
    X = E - E.mean(0); Xw = X @ np.linalg.inv(np.linalg.cholesky(np.cov(X.T))).T
    G = rng.standard_normal(Xw.shape)
    kurt = ((Xw ** 4).mean(0) - 3)
    print(f"rank {r}: whitened kurtosis per axis {np.round(kurt, 2)}")
    for k in (8, 9, 16):
        a_, l_, _ = kmeans(unit(Xw), k); b_, _, _ = kmeans(unit(G), k)
        print(f"  k={k:2d}: mean cos to centre real {a_:.4f}  gaussian {b_:.4f}  cluster sizes {np.sort(np.bincount(l_))[::-1][:9]}")
W = np.load(ROOT / "data/cache/38_vectors_c7_s0.npz")["W_test"].astype(np.float64)
_, Ue = np.linalg.eigh(W[1:].T @ W[1:]); P = Ue[:, ::-1][:, :8]
n = min(len(W), len(V)); g = good[good < n]
A = W[g] @ P; B = V[g, 1:10] / np.sqrt(di[g])[:, None]
A = A - A.mean(0); B = B - B.mean(0)
Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
print("canonical correlations, MF top-8 vs exposure dims 1-9:", np.round(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 3))
