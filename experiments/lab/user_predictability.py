# Run from the repository root: .venv/bin/python experiments/lab/user_predictability.py
# DO USERS DIFFER IN TASTE STRENGTH (CHOICE TEMPERATURE) BEYOND THEIR ACTIVITY? Real heavy users are more taste-driven than
# light ones (lab/replica_cells.py); LightGBM sees that through history length. A per-user temperature independent of
# activity would be invisible to it. Measure each user's PAST predictability on days 0-5: for every clicked event, the
# margin of the clicked item over the event's other 9 under the user's leave-one-out taste (projected 8-dim unit MF vectors
# of days 0-5, 1 seed) and the clicked item's rank percentile. Then on day 6 (clean: not in the MF's training data), within
# each known-clicks band, the projected-cosine AUC by tercile of past predictability, and the same by tercile of coherence
# (norm of the mean unit vector, a feature-like proxy the model can already form). A large spread within a band, larger
# than coherence's, means a per-user temperature feature is missing.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
auc = e13.roc_auc
C = ROOT / "data/cache"
fit, _, _, _, _ = e13.load_base(str(C / "13_base_c7_9c47a117c762.npz"))
def unit(A): return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
W = np.load(C / "22_vectors_c7.npz")["W_6"].astype(np.float64)
_, U = np.linalg.eigh(W[1:].T @ W[1:]); Z = unit(W @ U[:, ::-1][:, :8]); Z[0] = 0
items = fit["item"].reshape(-1, 10).astype(np.int64); lab = fit["label"].reshape(-1, 10); eu = fit["user"].astype(np.int64)
NU = int(max(eu.max(), fit["clk_user"].max())) + 1
k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
P = np.zeros((NU, 8)); np.add.at(P, cu, Z[ci]); npos = np.bincount(cu, minlength=NU)
# past predictability over clicked events of days 0-5 (leave the event's own click out of the taste)
tr = np.flatnonzero((fit["day"] < 6) & (lab.max(1) > 0))
c = items[tr][lab[tr] > 0]; u = eu[tr]
th = unit(P[u] - Z[c])
cs = np.einsum("ed,ekd->ek", th, Z[items[tr]])
cc = np.einsum("ed,ed->e", th, Z[c])
margin = cc - (cs.sum(1) - cc) / 9
pct = ((cs < cc[:, None]).sum(1)) / 9.0
ok = npos[u] >= 3
cnt = np.bincount(u[ok], minlength=NU)
mg = np.bincount(u[ok], margin[ok], NU) / np.maximum(cnt, 1)
pc = np.bincount(u[ok], pct[ok], NU) / np.maximum(cnt, 1)
coh = np.linalg.norm(P, axis=1) / np.maximum(npos, 1)
print(f"clicked events days 0-5: {ok.sum()}, mean rank percentile {pct[ok].mean():.3f}, mean margin {margin[ok].mean():.4f}")
# split-half reliability of the per-user margin (users with >= 10 scored events)
par = np.random.default_rng(0).random(ok.sum()) < 0.5
uu, mm_ = u[ok], margin[ok]
a = np.bincount(uu[par], mm_[par], NU) / np.maximum(np.bincount(uu[par], minlength=NU), 1)
b = np.bincount(uu[~par], mm_[~par], NU) / np.maximum(np.bincount(uu[~par], minlength=NU), 1)
for lo, hi in ((10, 20), (21, 50), (51, 10**6)):
    s = (cnt >= lo) & (cnt <= hi)
    print(f"  users with {lo}-{hi} scored events: {s.sum()}, split-half corr of margin {np.corrcoef(a[s], b[s])[0, 1]:.3f}, "
          f"corr(margin, coherence) {np.corrcoef(mg[s], coh[s])[0, 1]:.3f}", flush=True)
# day 6
H, n = e13.long_history(fit, fit, 100)
d6 = np.flatnonzero((fit["day"] == 6) & (n > 0))
h, nn = H[d6].astype(np.int64), n[d6].astype(int)
msk = (np.arange(100)[None] < nn[:, None])
S = (Z[h] * msk[:, :, None]).sum(1)
s6 = np.einsum("ed,ekd->ek", unit(S), Z[items[d6]])
y6 = lab[d6]; u6 = eu[d6]
for lo, hi in ((6, 20), (21, 50), (51, 100)):
    band = (nn >= lo) & (nn <= hi) & (cnt[u6] >= 3)
    for name, stat in (("predictability (margin)", mg), ("rank percentile", pc), ("coherence", coh)):
        v = stat[u6][band]; q = np.quantile(v, [1 / 3, 2 / 3])
        out = []
        for t, sel in enumerate((v <= q[0], (v > q[0]) & (v <= q[1]), v > q[1])):
            ss, yy = s6[band][sel].ravel(), y6[band][sel].ravel()
            within = np.mean([auc(a_, b_) for a_, b_ in zip(s6[band][sel][:4000], y6[band][sel][:4000]) if 0 < b_.sum() < 10]) \
                if False else None
            out.append(f"T{t + 1} {auc(ss, yy):.4f} (ctr {yy.mean():.4f})")
        print(f"day 6, known clicks {lo}-{hi} ({band.sum()} events), by {name:24s}: " + "  ".join(out), flush=True)
