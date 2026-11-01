# Run from the repository root: .venv/bin/python experiments/lab/shown_profile.py
# Needs data/cache/22_vectors_c7.npz (MF vectors of an MF trained on days < d, 1 seed).
#
# Is the user's IMPRESSION history (every item shown to them in earlier events, hundreds for active
# users who click little) a click-free estimate of their taste? Candidates are mildly personalised
# (RESULTS.md: cosine to the history 0.034 vs -0.001), which, averaged over many impressions, could
# be strong for exactly the light-click users the evaluation weighs most. Day 6 (vectors of the MF
# trained on days 0-5): score = candidate . (mean vector of the items shown to the user before day 6
# - the mean over all impressions), alone and next to the MF dot, overall and by known clicks and by
# number of earlier impressions.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, n_users, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(ROOT / "data/cache/22_vectors_c7.npz")
W = z["W_6"].astype(np.float64); evd = z["ev_6"]; hv = z["hv_6"].astype(np.float64)
auc = e13.roc_auc
before = np.flatnonzero(fit["day"] < 6)
su = np.repeat(fit["user"][before].astype(np.int64), 10)
si = fit["item"].reshape(-1, 10)[before].ravel()
n_u = int(fit["user"].max()) + 1
S = np.zeros((n_u, W.shape[1])); np.add.at(S, su, W[si])
ns = np.bincount(su, minlength=n_u).astype(float)
g = W[si].mean(0)
prof = S / np.maximum(ns, 1)[:, None] - g * (ns > 0)[:, None]
items = fit["item"].reshape(-1, 10)[evd]
y = fit["label"].reshape(-1, 10)[evd].ravel()
u = fit["user"][evd]
s_shown = np.einsum("ed,ekd->ek", prof[u], W[items]).ravel()
s_mf = np.einsum("ed,ekd->ek", hv, W[items]).ravel()
_, hl = e13.long_history(fit, fit, e13.HIST_LONG)
h = np.repeat(hl[evd], 10); nimp = np.repeat(ns[u], 10)
zs = lambda s: (s - s.mean()) / s.std()
print(f"day 6: shown-profile score alone AUC {auc(s_shown, y):.5f}; MF dot alone {auc(s_mf, y):.5f}; "
      f"corr {np.corrcoef(s_shown, s_mf)[0, 1]:.3f}")
for w in [0.1, 0.3, 0.5]:
    print(f"  z(MF dot) + {w} z(shown): {auc(zs(s_mf) + w * zs(s_shown), y):.5f}")
print("by known clicks (shown alone / MF dot / MF + 0.3 shown):")
for lo, hi in [(0, 0), (1, 5), (6, 20), (21, 100)]:
    k = (h >= lo) & (h <= hi)
    print(f"  {lo:3d}-{hi:3d} ({k.mean():.1%} of rows, median earlier impressions {np.median(nimp[k]):.0f}): "
          f"{auc(s_shown[k], y[k]):.4f} / {auc(s_mf[k], y[k]):.4f} / {auc(zs(s_mf)[k] + 0.3 * zs(s_shown)[k], y[k]):.4f}")
# heavy users: do the shown profile and the click profile point the same way?
mfu = np.zeros((n_u, W.shape[1])); mfu[u] = hv
heavy = (ns > 200) & (np.linalg.norm(mfu, axis=1) > 0)
cos = (prof[heavy] * mfu[heavy]).sum(1) / (np.linalg.norm(prof[heavy], axis=1) * np.linalg.norm(mfu[heavy], axis=1) + 1e-12)
print(f"users with >200 earlier impressions: cosine(shown profile, click profile) mean {cos.mean():.3f}, "
      f"median {np.median(cos):.3f}")
