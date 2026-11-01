# Run from the repository root: .venv/bin/python experiments/lab/ppmi_cos.py
# A DIFFERENT ITEM ESTIMATOR under today's scoring (2026-10-30): co-click PPMI + truncated SVD item vectors (26
# vectors_pool.ppmi_svd_vectors, clicks before day 6), scored like the core view: projected (top-k) unit vectors, user = mean of
# the known clicks, cosine. Earlier PPMI / PureSVD / EASE checks used the dot product, where popularity swamped taste. Day-6
# single-score AUC (plain / eval-weighted) and rank correlation with the MF's projected cosine (38_vectors seed 0).
import importlib.util
import numpy as np
from scipy.stats import rankdata
spec = importlib.util.spec_from_file_location("vp", "experiments/26_audience/vectors_pool.py")
vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
e13 = vp
fit, test, _, _, n_items = vp.load_base("data/cache/13_base_c7_9c47a117c762.npz")
D = 6
evd = np.flatnonzero(fit["day"] == D); it = fit["item"].reshape(-1, 10)[evd].astype(np.int64); y = fit["label"].reshape(-1, 10)[evd].ravel()
H, n = vp.long_history(fit, fit, 100); H, n = H[evd].astype(np.int64), n[evd].astype(int)
m = (np.arange(100)[None] < n[:, None])[:, :, None]
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
def proj(W, k):
    W = np.nan_to_num(W.astype(np.float64)); _, U = np.linalg.eigh(W[1:].T @ W[1:]); Z = unit(W @ U[:, ::-1][:, :k]); Z[0] = 0
    return Z
def score(Z):
    hv = unit((Z[H] * m).sum(1)); return np.einsum("ed,ekd->ek", hv, Z[it]).ravel()
mf = score(proj(np.load("data/cache/38_vectors_c7_s0.npz")[f"W_{D}"], 8))
print(f"MF projected cosine (1 seed): {vp.roc_auc(mf, y):.5f}", flush=True)
for dim in (32, 64):
    V = vp.ppmi_svd_vectors(fit, D, n_items + 1, dim)
    for k in (8, 12, dim):
        s = score(proj(V, k))
        print(f"PPMI-SVD dim {dim}, top {k}: {vp.roc_auc(s, y):.5f}  rank corr with MF {np.corrcoef(rankdata(s), rankdata(mf))[0, 1]:.3f}  "
              f"MF + PPMI (rank mean) {vp.roc_auc(rankdata(s) + rankdata(mf), y):.5f}", flush=True)
