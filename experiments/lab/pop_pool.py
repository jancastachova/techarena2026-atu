# Run from the repository root: .venv/bin/python experiments/lab/pop_pool.py
# POPULARITY-WEIGHTED POOLING OF THE USER'S CLICKS. The calibrated replica (lab/replica_cells.py) trails the real data for
# very popular candidates (101+ clicks: real projected-cosine AUC +0.022 to +0.026 for users <= 50 clicks): popular items
# act more taste-specific than the MF says. The IDF views weight a user's clicks by pop^-a (a > 0); the opposite direction
# (clicks on popular items weighted UP) was never tried. Day 6, users with >= 1 known click, MF vectors of days 0-5
# (3 seeds), projected 8-dim unit vectors; taste = unit(sum_h w_h z_h), w_h = (clicks of h before day 6 + 1)^a.
# Reported: projected-cosine AUC overall, by known clicks and by candidate popularity, and the rank correlation with a = 0.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
auc = e13.roc_auc
C = ROOT / "data/cache"
fit, _, _, _, _ = e13.load_base(str(C / "13_base_c7_9c47a117c762.npz"))
H, n = e13.long_history(fit, fit, 100)
d6 = np.flatnonzero((fit["day"] == 6) & (n > 0))
h, nn = H[d6].astype(np.int64), n[d6].astype(int)
m = (np.arange(100)[None] < nn[:, None])
it = fit["item"].reshape(-1, 10)[d6].astype(np.int64)
y = fit["label"].reshape(-1, 10)[d6].ravel()
NI = int(max(fit["item"].max(), fit["clk_item"].max())) + 1
k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
clk = np.bincount(fit["clk_item"][k], minlength=NI)
def unit(A): return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
Zs = []
for v in [C / "22_vectors_c7.npz", C / "26_vectors_c7_s1.npz", C / "26_vectors_c7_s2.npz"]:
    W = np.load(v)["W_6"].astype(np.float64)
    _, U = np.linalg.eigh(W[1:].T @ W[1:]); Z = unit(W @ U[:, ::-1][:, :8]); Z[0] = 0
    Zs.append(Z)
kn = np.repeat(nn, 10); cp = clk[it].ravel()
GR = [("1-5", (kn <= 5)), ("6-20", (kn > 5) & (kn <= 20)), ("21+", kn > 20),
      ("cand 0-10", cp <= 10), ("cand 11-100", (cp > 10) & (cp <= 100)), ("cand 101+", cp > 100)]
from scipy.stats import spearmanr
base = None
for a in (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
    w = ((clk[h] + 1.0) ** a) * m
    s = 0
    for Z in Zs:
        S = (Z[h] * w[:, :, None]).sum(1)
        s = s + np.einsum("ed,ekd->ek", unit(S), Z[it]).ravel() / len(Zs)
    if a == 0.0: base = s
    rc = "" if base is None else f"  rank corr {spearmanr(s[::7], base[::7])[0]:.4f}"
    print(f"a {a:+.2f}: AUC {auc(s, y):.5f} | " + "  ".join(f"{g} {auc(s[mm], y[mm]):.4f}" for g, mm in GR) + rc, flush=True)
