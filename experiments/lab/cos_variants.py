# Run from the repository root: .venv/bin/python experiments/lab/cos_variants.py
# Which NORMALISED taste match is strongest? Day 6, the 10 saved vector seeds of 22 / 26 (MF of days 0-5), scores
# averaged over seeds; rank corr with the plain cosine (34's view). User pooling: count mean (34), sqrt-set
# (33), idf (1+clicks)^-1, mean of UNIT item vectors; candidate unit vector in all cases.
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
files = [ROOT / "data/cache/22_vectors_c7.npz"] + [ROOT / f"data/cache/26_vectors_c7_s{s}.npz" for s in range(1, 10)]
evd = np.flatnonzero(fit["day"] == 6)
items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
H, n = e13.long_history(fit, fit, 100); H, n = H[evd], n[evd].astype(int)
m = (np.arange(100)[None] < n[:, None]).astype(np.float32)
same = (H[:, :, None] == H[:, None, :]) & (m[:, None, :] > 0)
w_sq = m / np.sqrt(np.maximum(same.sum(2), 1))
k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
clk = np.bincount(fit["clk_item"][k], minlength=n_items)
w_idf = m / (1.0 + clk[H])
W_ = {"count": m, "sqrtset": w_sq, "idf1": w_idf, "idf.5": m / np.sqrt(1.0 + clk[H])}
acc = {}
for f in files:
    W = np.load(f)["W_6"].astype(np.float32)
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    Vn = Wn[items]
    for name, w in W_.items():
        for unit_hist in (False, True):
            src = Wn if unit_hist else W
            hv = (src[H] * w[:, :, None]).sum(1) / np.maximum(w.sum(1), 1e-9)[:, None]
            hv /= np.maximum(np.linalg.norm(hv, axis=1, keepdims=True), 1e-9)
            key = f"{name}{' unit-hist' if unit_hist else ''}"
            acc[key] = acc.get(key, 0) + np.einsum("ed,ekd->ek", hv, Vn).ravel() / len(files)
ref = rankdata(acc["count"])
h = np.repeat(n, 10); G = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21+", h >= 21)]
for key, s in acc.items():
    print(f"{key:20s} AUC {auc(s, y):.5f}  " + "  ".join(f"{g} {auc(s[mm], y[mm]):.4f}" for g, mm in G)
          + f"  rank corr with count cosine {np.corrcoef(ref, rankdata(s))[0, 1]:.4f}", flush=True)
