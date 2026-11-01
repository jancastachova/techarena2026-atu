# Run from the repository root: .venv/bin/python experiments/lab/mild_decay.py
# Needs data/cache/22_vectors_c7.npz (MF vectors of the MF trained on days < d, 1 seed).
#
# Published recommender simulators often move a user's taste a fixed fraction toward each clicked item
# (arXiv 2403.07571), i.e. taste = an EXPONENTIALLY weighted mean of the clicked items, recent clicks
# slightly heavier. RESULTS.md tested only STRONG recency (last 5 / 10 clicks, decay constants 5 and
# 20 clicks: all worse) and concluded "no recency". A MILD decay was never tested. Day 6: user vector
# = sum w_j v_j / sum w_j over the known clicks (newest first, up to 100), w_j = exp(-j / tau), by
# known clicks; tau = inf is the plain mean. Also a decay by TIME (hours before the event).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
z = np.load(ROOT / "data/cache/22_vectors_c7.npz")
W = z["W_6"].astype(np.float64); evd = z["ev_6"]
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
H = hist[evd]; n = hlen[evd].astype(int); m = np.arange(H.shape[1])[None] < n[:, None]
items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
V = W[H]                                                   # (E, 100, d), newest first
h = np.repeat(n, 10)
groups = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21-50", (h >= 21) & (h <= 50)), ("51+", h >= 51)]
def score(wts):
    wts = wts * m
    hv = (V * wts[:, :, None]).sum(1) / np.maximum(wts.sum(1), 1e-12)[:, None]
    return np.einsum("ed,ekd->ek", hv, W[items]).ravel()
pos = np.arange(H.shape[1])[None].astype(float)
print("decay by click position (newest = 0):")
for tau in [10, 30, 50, 100, 200, 1000, np.inf]:
    s = score(np.exp(-pos / tau) if np.isfinite(tau) else np.ones_like(pos))
    print(f"  tau {tau:>6}: all {auc(s, y):.5f}  " + "  ".join(f"{g} {auc(s[k], y[k]):.4f}" for g, k in groups), flush=True)
# time decay: hours between each history click and the event
u, it, ts = fit["clk_user"].astype(np.int64), fit["clk_item"], fit["clk_ts"].astype(np.int64)
o = np.lexsort((it, ts, u)); key = u[o] * 10_000_000_000 + ts[o]; tso = ts[o]
eu = fit["user"][evd].astype(np.int64); et = fit["ts"][evd].astype(np.int64)
end = np.searchsorted(key, eu * 10_000_000_000 + et, side="left")
HT = np.zeros(H.shape)
for j in range(H.shape[1]):
    idx = end - 1 - j; ok = j < n
    HT[ok, j] = (et[ok] - tso[idx[ok]]) / 3600
print("decay by time (hours before the event):")
for th in [24, 72, 168, 500]:
    s = score(np.exp(-HT / th))
    print(f"  half-life-ish {th:>4} h: all {auc(s, y):.5f}  " + "  ".join(f"{g} {auc(s[k], y[k]):.4f}" for g, k in groups), flush=True)
