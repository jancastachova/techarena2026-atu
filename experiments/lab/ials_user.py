# Run from the repository root: .venv/bin/python experiments/lab/ials_user.py [--days 6]
# Needs data/cache/22_vectors_c7.npz (item vectors W_<d> of an MF trained on days < d, 1 seed).
#
# A different ESTIMATOR of the user's taste from the same item vectors. The pipeline's user vector
# is the mean of the vectors of the user's known clicks; IDF pooling (25) showed that how the
# clicks are pooled carries information the mean misses. Here the user vector is the implicit-ALS
# solve given fixed item vectors:
#     u = (G + a/n * H'H + lam*I)^-1 (a/n * H'1),   G = sum_i p_i w_i w_i'  (p = impression share)
# H = vectors of the user's known clicks (up to 100), n = their count. Unlike the mean it
# discounts redundant clicks (many clicks in one direction count once), contrasts the clicks with
# what users are typically shown (G), and shrinks light users toward zero. Screening: day-6 AUC
# of the dot product alone vs the mean-pooled dot, and the rank correlation between them.
import argparse, importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
ap = argparse.ArgumentParser()
ap.add_argument("--days", default="6")
ap.add_argument("--grid", default="1:0.01,10:0.01,10:0.1,100:0.1,100:1,1000:1")
a = ap.parse_args()
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(str(ROOT / "data/cache/22_vectors_c7.npz"))
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
auc = e13.roc_auc


def ials_users(W, H, n, p, a_, lam):
    """Per event: implicit-ALS user vector given item vectors W, history codes H (0 = pad)."""
    G = (W * p[:, None]).T @ W
    V = W[H] * (np.arange(H.shape[1])[None, :] < n[:, None])[:, :, None]   # (E, K, D)
    nn = np.maximum(n, 1)[:, None, None].astype(np.float32)
    A = G[None] + a_ / nn * np.einsum("ekd,ekf->edf", V, V) + lam * np.trace(G) / W.shape[1] * np.eye(W.shape[1])[None]
    b = a_ / nn[:, :, 0] * V.sum(1)
    return np.linalg.solve(A, b[:, :, None])[:, :, 0]


for day in [int(x) for x in a.days.split(",")]:
    W = z[f"W_{day}"].astype(np.float64); evd = z[f"ev_{day}"]
    rows_before = np.repeat(fit["day"], 10) < day
    p = np.bincount(fit["item"][rows_before], minlength=W.shape[0]).astype(np.float64); p /= p.sum()
    H, n = hist[evd], hlen[evd].astype(np.int64)
    items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
    q = W[items]                                                           # (E, 10, D)
    m = (np.arange(H.shape[1])[None, :] < n[:, None])
    mean_u = (W[H] * m[:, :, None]).sum(1) / np.maximum(n, 1)[:, None]
    s_mean = np.einsum("ed,ekd->ek", mean_u, q).ravel()
    hv = z[f"hv_{day}"].astype(np.float64)
    print(f"day {day}: {len(evd):,} events; mean-pool dot AUC {auc(s_mean, y):.5f} "
          f"(cached hv dot {auc(np.einsum('ed,ekd->ek', hv, q).ravel(), y):.5f})", flush=True)
    r_mean = rankdata(s_mean)
    for spec_ in a.grid.split(","):
        a_, lam = (float(v) for v in spec_.split(":"))
        u = ials_users(W, H, n, p, a_, lam)
        s = np.einsum("ed,ekd->ek", u, q).ravel()
        print(f"  iALS a={a_:<6g} lam={lam:<5g} dot AUC {auc(s, y):.5f}  rank corr with mean {np.corrcoef(rankdata(s), r_mean)[0, 1]:.4f}", flush=True)
