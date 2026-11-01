# Run from the repository root: .venv/bin/python experiments/lab/set_alpha.py
# Needs data/cache/22_vectors_c7.npz and 26_vectors_c7_s1..4.npz. Set pooling sweep: each unique clicked
# item weighted by (times clicked)^alpha (alpha 1 = the plain per-click mean, 0 = unique items), day 6,
# MF vectors of days 0-5, taste match averaged over 5 vector seeds; by known clicks.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
files = [ROOT / "data/cache/22_vectors_c7.npz"] + [ROOT / f"data/cache/26_vectors_c7_s{s}.npz" for s in range(1, 5)]
evd = np.load(files[0])["ev_6"]
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
H = hist[evd]; n = hlen[evd].astype(int); m = np.arange(H.shape[1])[None] < n[:, None]
cnt = np.maximum(((H[:, :, None] == H[:, None, :]) & m[:, None, :]).sum(2), 1)
items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
h = np.repeat(n, 10)
groups = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21-50", (h >= 21) & (h <= 50)), ("51+", h >= 51)]
Ws = [np.load(f)["W_6"].astype(np.float32) for f in files]
for alpha in [1.0, 0.75, 0.5, 0.25, 0.0, -0.25]:
    w = m * cnt ** (alpha - 1.0)                       # per click: count^(alpha-1) -> per unique item count^alpha
    s = 0
    for W in Ws:
        hv = (W[H] * w[:, :, None]).sum(1) / np.maximum(w.sum(1), 1e-9)[:, None]
        s = s + np.einsum("ed,ekd->ek", hv, W[items]).ravel() / len(Ws)
    print(f"alpha {alpha:+.2f}: all {auc(s, y):.5f}  " + "  ".join(f"{g} {auc(s[k], y[k]):.4f}" for g, k in groups), flush=True)
