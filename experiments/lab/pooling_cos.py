# Run from the repository root: .venv/bin/python experiments/lab/pooling_cos.py
# How should a user's clicks be POOLED in cosine space? 34/35 score cos(mean of the clicked vectors, candidate). Alternatives:
# the mean of the per-click cosines, the max, and a soft-max (tau x log-mean-exp of tau-scaled cosines). The dot-product
# max ("multi-interest", lab/multi_interest.py) failed when norms were still noise. Day 6, 5 vector seeds, projected (k=8)
# vectors and plain vectors, by known clicks.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
files = [ROOT / "data/cache/22_vectors_c7.npz"] + [ROOT / f"data/cache/26_vectors_c7_s{s}.npz" for s in range(1, 5)]
evd = np.flatnonzero(fit["day"] == 6); items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
H, n = e13.long_history(fit, fit, 100); H, n = H[evd], n[evd].astype(int)
m = np.arange(100)[None] < n[:, None]
unit = lambda X: X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
acc = {}
for f in files:
    for proj in (False, True):
        W = np.load(f)["W_6"].astype(np.float64)
        if proj:
            _, U = np.linalg.eigh(W[1:].T @ W[1:]); U = U[:, ::-1][:, :8]; W = (W @ U) @ U.T
        Wn = unit(W); Wn[0] = 0
        tag = "proj8" if proj else "plain"
        hv = unit((Wn[H] * m[:, :, None]).sum(1))
        acc[(tag, "cos of mean")] = acc.get((tag, "cos of mean"), 0) + np.einsum("ed,ekd->ek", hv, Wn[items]).ravel() / len(files)
        for s in range(0, len(evd), 4000):
            sl = slice(s, s + 4000)
            C = np.einsum("ehd,ekd->ekh", Wn[H[sl]], Wn[items[sl]])           # (e, 10, 100) per-click cosines
            mk = m[sl][:, None, :]
            nn = np.maximum(n[sl], 1)[:, None]
            res = {"mean of cosines": np.where(mk, C, 0).sum(2) / nn, "max cosine": np.where(mk, C, -2).max(2)}
            for tau in (5.0, 10.0):
                e_ = np.where(mk, np.exp(tau * C), 0).sum(2) / nn
                res[f"softmax tau {tau:g}"] = np.log(np.maximum(e_, 1e-300)) / tau
            for k_, v_ in res.items():
                key = (tag, k_)
                if key not in acc:
                    acc[key] = np.zeros(len(y))
                acc[key][s * 10:(s + 4000) * 10] += np.where(np.repeat(n[sl], 10) > 0, v_.ravel(), 0) / len(files)
h = np.repeat(n, 10); G = [("1-2", (h >= 1) & (h <= 2)), ("3-5", (h >= 3) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21+", h >= 21)]
for (tag, k_), s in acc.items():
    print(f"{tag:5s} {k_:18s} AUC {auc(s, y):.5f}  " + "  ".join(f"{g} {auc(s[mm], y[mm]):.4f}" for g, mm in G), flush=True)
