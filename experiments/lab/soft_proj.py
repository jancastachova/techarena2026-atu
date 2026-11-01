# Run from the repository root: .venv/bin/python experiments/lab/soft_proj.py
# SOFT projection: the hard cut onto the 8 strong principal directions (lab/geometry.py, the #1 feature) drops the medium
# directions for every item, popular ones included. Here each item keeps its top-8 component and its remaining components
# scaled by c / (c + K), c = the item's clicks before the day: rare items are shrunk into the 8-dim subspace, popular
# items keep their detail. Day 6, 10 vector seeds, cosine with the count-pooled user (built from the shrunk vectors).
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
m = (np.arange(100)[None] < n[:, None]).astype(np.float32)
k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
clk = np.bincount(fit["clk_item"][k], minlength=n_items + 1).astype(np.float64)
cpop = clk[items].ravel()
def cos_score(W):
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    hv = (W[H] * m[:, :, None]).sum(1) / np.maximum(n, 1)[:, None]
    hv /= np.maximum(np.linalg.norm(hv, axis=1, keepdims=True), 1e-9)
    return np.einsum("ed,ekd->ek", hv, Wn[items]).ravel()
acc = {}
for f in files:
    W = np.load(f)["W_6"].astype(np.float64)
    _, U = np.linalg.eigh(W[1:].T @ W[1:]); U8 = U[:, ::-1][:, :8]
    top = (W @ U8) @ U8.T; rest = W - top
    var = {"plain": W, "hard k=8": top}
    for K in (2, 5, 10, 20, 50, 200):
        lam = (clk[:len(W)] / (clk[:len(W)] + K))[:, None]
        var[f"soft K={K}"] = top + lam * rest
    for kk, V in var.items():
        acc[kk] = acc.get(kk, 0) + cos_score(V) / len(files)
PB = [("0-3", cpop <= 3), ("4-10", (cpop >= 4) & (cpop <= 10)), ("11-30", (cpop >= 11) & (cpop <= 30)), ("31+", cpop >= 31)]
h = np.repeat(n, 10); G = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21+", h >= 21)]
for kk, s in acc.items():
    print(f"{kk:12s} AUC {auc(s, y):.5f} | " + "  ".join(f"{g} {auc(s[mm], y[mm]):.4f}" for g, mm in G) +
          " | cand clicks " + "  ".join(f"{g} {auc(s[mm], y[mm]):.4f}" for g, mm in PB), flush=True)
