# Run from the repository root: .venv/bin/python experiments/lab/simplex_test.py
# ARE THE 8 EQUAL DIRECTIONS 8 CATEGORIES? Eight equal eigenvalues fit both an isotropic Gaussian cloud and a symmetric
# simplex (items near 8 corners, or non-negative mixtures of 8 topics). Day-6 item vectors of 10 seeds (W_6), projected on
# the top 8 directions; well-measured items (>= 31 clicks before day 6). Compared with a Gaussian of the same covariance:
#   1. k-means (k = 8) on unit vectors: mean cosine to the nearest centre
#   2. the most non-Gaussian 1-d projection (FastICA-style kurtosis search from 200 random starts): max |excess kurtosis|
#   3. if corners exist: share of items whose top corner weight > 0.7 (non-negative least squares on the ICA corners)
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
rng = np.random.default_rng(0)
W = np.load(ROOT / "data/cache/22_vectors_c7.npz")["W_6"].astype(np.float64)
NI = len(W)
k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
clicks = np.bincount(fit["clk_item"][k], minlength=NI)
good = np.flatnonzero(clicks[:NI] >= 31); good = good[good > 0]
_, U = np.linalg.eigh(W[1:].T @ W[1:]); Z = W @ U[:, ::-1][:, :8]
X = Z[good]; X = X - X.mean(0)
L = np.linalg.cholesky(np.cov(X.T)); Xw = X @ np.linalg.inv(L).T            # whitened
G = rng.standard_normal(Xw.shape)
def unit(A): return A / np.linalg.norm(A, axis=1, keepdims=True)
def kmeans_fit(A, k=8, it=50, starts=5):
    best = -1
    for s in range(starts):
        C = A[rng.choice(len(A), k, replace=False)]
        for _ in range(it):
            a = np.argmax(A @ C.T, 1)
            C = unit(np.stack([A[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)]))
        best = max(best, np.max(A @ C.T, 1).mean())
    return best
def kurt_search(A, starts=200):
    best = 0
    for _ in range(starts):
        w = unit(rng.standard_normal((1, 8)))[0]
        for _ in range(100):                                              # FastICA one-unit, kurtosis contrast
            s = A @ w; w = unit(((A * (s ** 3)[:, None]).mean(0) - 3 * w)[None])[0]
        s = A @ w; best = max(best, abs(np.mean(s ** 4) - 3))
    return best
print(f"{len(good)} items with >= 31 clicks")
for name, A in (("real (whitened)", Xw), ("gaussian", G)):
    print(f"{name:16s} k-means(8) mean cos to centre {kmeans_fit(unit(A)):.4f}   max |excess kurtosis| {kurt_search(A):.3f}"
          f"   mean |excess kurtosis| of the 8 PCA axes {np.mean(np.abs((A ** 4).mean(0) - 3)):.3f}", flush=True)
