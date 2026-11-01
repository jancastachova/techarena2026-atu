# Run from the repository root: .venv/bin/python experiments/lab/multi_interest.py
# Needs data/cache/22_vectors_c7.npz (MF vectors of an MF trained on days < d, 1 seed).
#
# Multi-interest check (k-max latent matching): for day-6 users with >= 6 known clicks, split the
# clicked items' MF vectors into k = 2 clusters (k-means, 10 iterations, deterministic init from the
# two most distant items) and score a candidate by max(u1 . v, u2 . v), vs the single mean u . v.
# Also the softmax-weighted (soft max) version. RESULTS.md found taste to be one direction (candidate-
# aware attention 0.7837 vs 0.7856); this is the explicit k-means version.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(ROOT / "data/cache/22_vectors_c7.npz")
W = z["W_6"].astype(np.float64); evd = z["ev_6"]
auc = e13.roc_auc
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
n = hlen[evd].astype(int)
mean_s = np.zeros((len(evd), 10)); max_s = np.zeros((len(evd), 10)); soft_s = np.zeros((len(evd), 10))
for i, e in enumerate(evd):
    V = W[hist[e, :n[i]]] if n[i] > 0 else np.zeros((1, W.shape[1]))
    C = W[items[i]]
    mean_s[i] = C @ V.mean(0)
    if n[i] >= 6:
        d = ((V[:, None] - V[None]) ** 2).sum(2); a, b = np.unravel_index(d.argmax(), d.shape)
        cent = V[[a, b]].copy()
        for _ in range(10):
            lab = ((V[:, None] - cent[None]) ** 2).sum(2).argmin(1)
            for k in range(2):
                if (lab == k).any():
                    cent[k] = V[lab == k].mean(0)
        sc = C @ cent.T                                  # (10, 2)
        wk = np.bincount(lab, minlength=2) / len(lab)    # cluster sizes
        max_s[i] = sc.max(1)
        soft_s[i] = (sc * wk).sum(1) + 0.5 * (sc.max(1) - (sc * wk).sum(1))
    else:
        max_s[i] = soft_s[i] = mean_s[i]
m6 = np.repeat(n >= 6, 10)
for name, s in [("single mean u . v", mean_s), ("k=2 max(u1 . v, u2 . v)", max_s), ("k=2 halfway mean-max", soft_s)]:
    s = s.ravel()
    print(f"{name:26s} users with >= 6 clicks: AUC {auc(s[m6], y[m6]):.5f}   all users: {auc(s, y):.5f}", flush=True)
