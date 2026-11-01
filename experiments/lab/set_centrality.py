# Run from the repository root: .venv/bin/python experiments/lab/set_centrality.py
# Needs data/cache/22_vectors_c7.npz (MF vectors of an MF trained on days < d, 1 seed).
#
# The candidate generator leans toward the user's taste (DATASET.md: shown and click profiles agree at
# cosine 0.42). If an event mixes items picked near the user's taste with random fillers, the picked
# ones resemble EACH OTHER, so a candidate's similarity to the other 9 of its own event may flag them.
# (RESULTS.md's co-impression test pooled item pairs over all events; this is within one event.)
# Day 6: centrality = candidate . mean(other 9 candidates), raw and cosine; alone, within-event, and
# next to the MF dot, overall and by known clicks.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(ROOT / "data/cache/22_vectors_c7.npz")
W = z["W_6"].astype(np.float64); evd = z["ev_6"]; hv = z["hv_6"].astype(np.float64)
auc = e13.roc_auc
items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd]
V = W[items]                                                   # (E, 10, d)
others = (V.sum(1, keepdims=True) - V) / 9
cen = (V * others).sum(2)
Vn = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-12)
on = (Vn.sum(1, keepdims=True) - Vn) / 9
cen_cos = (Vn * on).sum(2)
mf = np.einsum("ed,ekd->ek", hv, V)
_, hl = e13.long_history(fit, fit, e13.HIST_LONG); h = hl[evd]
def within(s):
    has = y.max(1) > 0
    return np.mean([auc(s[i], y[i]) for i in np.flatnonzero(has)[:20000]])
zs = lambda s: (s - s.mean()) / s.std()
for name, s in [("centrality (dot)", cen), ("centrality (cosine)", cen_cos)]:
    print(f"{name}: global AUC {auc(s.ravel(), y.ravel()):.5f}, within-event {within(s):.4f}")
    for w in [0.05, 0.1, 0.2]:
        print(f"  z(MF dot) + {w} z({name}): {auc((zs(mf) + w * zs(s)).ravel(), y.ravel()):.5f}  (MF dot alone {auc(mf.ravel(), y.ravel()):.5f})")
for lo, hi in [(0, 0), (1, 5), (6, 20), (21, 100)]:
    k = (h >= lo) & (h <= hi)
    print(f"  known clicks {lo}-{hi}: centrality (cosine) alone {auc(cen_cos[k].ravel(), y[k].ravel()):.4f}")
