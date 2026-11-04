# Run from the repository root: .venv/bin/python experiments/lab/exposure_rank.py
# 2026-11-04: how many item groups does the generator have? Singular values of the normalised exposure matrix down to the
# noise floor (candidates shuffled across users gives ~0.1767), and for each k = rank + 1 the k-means tightness of well-shown
# items in the whitened embedding vs a Gaussian.
import importlib.util
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
u = np.repeat(fit["user"].astype(np.int64), 10); it = fit["item"].astype(np.int64); ts = np.repeat(fit["ts"], 10)
NI = int(max(n_items + 1, it.max() + 1, fit["clk_item"].max() + 1)); NU = int(max(fit["user"].max(), fit["clk_user"].max())) + 1
ck = fit["clk_user"].astype(np.int64) * NI + fit["clk_item"].astype(np.int64)
o = np.lexsort((fit["clk_ts"], ck)); ck_s, ct_s = ck[o], fit["clk_ts"][o]
f1 = np.r_[True, ck_s[1:] != ck_s[:-1]]; pk, pt = ck_s[f1], ct_s[f1]
key = u * NI + it; q = np.minimum(np.searchsorted(pk, key), len(pk) - 1)
keep = ~((pk[q] == key) & (pt[q] < ts)); u, it = u[keep], it[keep]
M = sp.csr_matrix((np.ones(len(u), np.float32), (u, it)), shape=(NU, NI))
du = np.asarray(M.sum(1)).ravel(); di = np.asarray(M.sum(0)).ravel()
Mn = sp.diags(1 / np.sqrt(np.maximum(du, 1))) @ M @ sp.diags(1 / np.sqrt(np.maximum(di, 1)))
R = 90
U, S, Vt = svds(Mn.astype(np.float64), k=R); o = np.argsort(-S); S, V = S[o], Vt[o].T
print("singular values:"); print(np.round(S, 3).tolist())
rng = np.random.default_rng(0)
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
good = np.flatnonzero(di >= 300); print(len(good), "items with >= 300 impressions")
for r in (8, 15, 23, 31, 39, 47, 63):
    E = V[good, 1:r + 1] / np.sqrt(di[good])[:, None]; X = E - E.mean(0)
    Xw = unit(X @ np.linalg.inv(np.linalg.cholesky(np.cov(X.T))).T)
    k = r + 1; best = -1
    for s in range(3):
        C = Xw[rng.choice(len(Xw), k, replace=False)]
        for _ in range(40):
            l = np.argmax(Xw @ C.T, 1)
            C = unit(np.stack([Xw[l == j].mean(0) if (l == j).any() else C[j] for j in range(k)]))
        v = np.max(Xw @ C.T, 1)
        if v.mean() > best: best, bl = v.mean(), l
    sz = np.sort(np.bincount(bl, minlength=k))
    print(f"rank {r:2d} k {k:2d}: mean cos {best:.4f}, sizes min {sz[0]} median {int(np.median(sz))} max {sz[-1]}", flush=True)
