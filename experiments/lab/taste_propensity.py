# Run from the repository root: .venv/bin/python experiments/lab/taste_propensity.py
# Does a user's TASTE predict their click propensity? For light users the click rate is mostly noise;
# if the simulator links propensity to taste (e.g. concentrated tastes click more), the mean vector of
# a few clicked items would estimate it. LightGBM never sees that vector, only its dot products with
# the candidates. Users with >= 20 events in days 0-5 (click rate well measured); features from their
# known clicks before day 6 with the MF vectors of days 0-5: the 32-dim mean vector, its norm, the mean
# cosine between clicked items (coherence), mean log popularity. 5-fold ridge R^2 against the
# split-half reliability of the click rate itself.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
W = np.load(ROOT / "data/cache/22_vectors_c7.npz")["W_6"].astype(np.float64)
CUT = e13.TRAIN_START + 6 * e13.DAY
ev = np.flatnonzero(fit["ts"] < CUT)
eu = fit["user"][ev].astype(np.int64); yl = fit["label"].reshape(-1, 10)[ev].max(1).astype(float)
nu = int(max(eu.max(), fit["clk_user"].max())) + 1
n_ev = np.bincount(eu, minlength=nu); rate = np.bincount(eu, yl, nu) / np.maximum(n_ev, 1)
o = np.lexsort((fit["ts"][ev], eu)); su, sy = eu[o], yl[o]
r_in = np.arange(len(su)) - np.searchsorted(su, su, "left"); odd = r_in % 2 == 1
r_odd = np.bincount(su[odd], sy[odd], nu) / np.maximum(np.bincount(su[odd], minlength=nu), 1)
r_even = np.bincount(su[~odd], sy[~odd], nu) / np.maximum(np.bincount(su[~odd], minlength=nu), 1)

k = fit["clk_ts"] < CUT
cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
pop = np.log1p(np.bincount(ci, minlength=n_items))
users = np.flatnonzero(n_ev >= 20)
o2 = np.argsort(cu, kind="stable"); cu, ci = cu[o2], ci[o2]
st = np.searchsorted(cu, users, "left"); en = np.searchsorted(cu, users, "right")
Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
X = []; keep = []
for j, u in enumerate(users):
    its = ci[st[j]:en[j]]
    if len(its) < 3:
        continue
    m = W[its].mean(0); mn = Wn[its].mean(0)
    X.append(np.r_[m, np.linalg.norm(m), np.linalg.norm(mn), pop[its].mean(), np.log(len(its))]); keep.append(u)
X = np.array(X); keep = np.array(keep)
names = ["mean vector (32)", "norm", "coherence", "log popularity", "log known clicks"]
def ridge_r2(Xm, yv, lam=10.0, folds=5):
    rs = np.random.default_rng(0); f = rs.integers(0, folds, len(yv)); pred = np.zeros(len(yv))
    Xm = (Xm - Xm.mean(0)) / np.maximum(Xm.std(0), 1e-9)
    for q in range(folds):
        a, b = f != q, f == q
        A = np.c_[Xm[a], np.ones(a.sum())]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yv[a])
        pred[b] = np.c_[Xm[b], np.ones(b.sum())] @ w
    return 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum(), pred
y_ = rate[keep]
print(f"{len(keep)} users with >= 20 events before day 6; click rate mean {y_.mean():.3f} sd {y_.std():.3f}")
rel = np.corrcoef(r_odd[keep], r_even[keep])[0, 1]
print(f"split-half reliability of the click rate: {rel:.3f} (Spearman-Brown full: {2 * rel / (1 + rel):.3f})")
for cols, nm in [(slice(0, 32), "mean vector"), (slice(32, 33), "norm"), (slice(33, 34), "coherence"),
                 (slice(34, 35), "log popularity"), (slice(35, 36), "log known clicks"), (slice(0, 36), "all")]:
    r2, pred = ridge_r2(X[:, cols], y_)
    print(f"  {nm:18s} CV R^2 {r2:+.4f}  corr {np.corrcoef(pred, y_)[0, 1]:+.3f}")
