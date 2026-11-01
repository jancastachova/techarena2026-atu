# Run from the repository root: .venv/bin/python experiments/lab/score_form.py
# Is the click score really a DOT product? The replica (built with a dot product) cannot show a misspecified score
# function; real clicks are MORE coherent than the calibrated replica's (lab/replica_ppc.py) while our MF scores the
# same on both, which a wrong functional form would explain. Day 6, MF vectors of days 0-5 (22_vectors W_6, one
# seed), user = mean of known clicks: dot vs cosine vs item-normalised vs user-normalised vs -distance, alone and by
# known clicks; within-event AUC too (forms differ mostly in scale across events).
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
items = fit["item"].reshape(-1, 10)[evd]; Y = fit["label"].reshape(-1, 10)[evd]; y = Y.ravel()
H, n = e13.long_history(fit, fit, 100); H, n = H[evd], n[evd].astype(int)
mask = np.arange(100)[None, :] < n[:, None]
hv = (W[H] * mask[..., None]).sum(1) / np.maximum(n, 1)[:, None]
Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
hvn_items = (Wn[H] * mask[..., None]).sum(1) / np.maximum(n, 1)[:, None]      # mean of unit item vectors
V = W[items]; Vn = Wn[items]
nh = np.linalg.norm(hv, axis=1, keepdims=True)
forms = {
    "dot  u.v": np.einsum("ed,ekd->ek", hv, V),
    "item-normalised u.v/|v|": np.einsum("ed,ekd->ek", hv, Vn),
    "user-normalised u.v/|u|": np.einsum("ed,ekd->ek", hv / np.maximum(nh, 1e-9), V),
    "cosine": np.einsum("ed,ekd->ek", hv / np.maximum(nh, 1e-9), Vn),
    "unit-vector mean . unit v": np.einsum("ed,ekd->ek", hvn_items, Vn),
    "-distance^2": -((hv[:, None, :] - V) ** 2).sum(2),
}
h = np.repeat(n, 10)
G = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21+", h >= 21)]
clicked = np.flatnonzero(Y.max(1) > 0)
for name, S in forms.items():
    s = S.ravel()
    within = np.mean([auc(S[e], Y[e]) for e in clicked[:30000]])
    print(f"{name:28s} global {auc(s, y):.5f}  within-event {within:.4f}  " + "  ".join(f"{g} {auc(s[m], y[m]):.4f}" for g, m in G), flush=True)
