# Run from the repository root: .venv/bin/python experiments/lab/adversarial.py
# ADVERSARIAL VALIDATION: are any features computed differently for evaluation rows than for training rows?
# A pipeline inconsistency would leave the fit-period checks intact and cost only on the leaderboard. A LightGBM
# classifier separates evaluation rows from each user's FIRST day-6 event (the closest training analogue),
# events reweighted to the same known-clicks mix, on experiment 13's features with MF30. Features that separate
# the sets without a legitimate reason (the evaluation has longer gaps since the last event and stats frozen at
# the cutoff) point at a bug. Also: per-feature AUC (eval vs day 6) and quantiles.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
Z = [np.load(ROOT / "data/cache" / f) for f in ("13_mf_c7_bb1fcb0a9c89.npz", "15_mf_c7_46f30c940a6a.npz")]
mf_fit = np.mean([z["mf_fit"] for z in Z], 0); mf_test = np.mean([z["mf_test"] for z in Z], 0)
d6 = np.flatnonzero(fit["day"] == 6)
_, first = np.unique(fit["user"][d6], return_index=True); ev6 = d6[first]
rows6 = (ev6[:, None] * 10 + np.arange(10)).ravel()
_, X6, names = e13.feature_matrix(fit, mf_fit, rows6)
_, XT, _ = e13.feature_matrix(test, mf_test, np.arange(len(test["item"])))
_, hl6 = e13.long_history({"user": fit["user"][ev6], "ts": fit["ts"][ev6]}, fit, 100)
_, hlT = e13.long_history(test, fit, 100)
B = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def bk(h): return np.searchsorted([b[1] for b in B], np.asarray(h, int))
sT = np.bincount(bk(hlT), minlength=5) / len(hlT); s6 = np.bincount(bk(hl6), minlength=5) / len(hl6)
w6 = np.repeat((sT / np.maximum(s6, 1e-9))[bk(hl6)], 10)
X = np.concatenate([X6, XT]); y = np.r_[np.zeros(len(X6)), np.ones(len(XT))]; w = np.r_[w6, np.ones(len(XT))]
print(f"day-6 first events {len(ev6)}, evaluation events {len(hlT)}; {len(names)} features", flush=True)
print("per-feature AUC (evaluation vs reweighted day-6 first events), most separating first:")
def wauc(s, yy, ww):
    o = np.argsort(s, kind="mergesort"); s, yy, ww = s[o], yy[o], ww[o]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, ww * yy); wn = np.bincount(g, ww * (1 - yy)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
res = []
for j, nm in enumerate(names):
    s = np.nan_to_num(X[:, j], nan=-999)
    res.append((abs(wauc(s, y, w) - 0.5), nm, wauc(s, y, w),
                np.percentile(X6[:, j][~np.isnan(X6[:, j])], [10, 50, 90]), np.percentile(XT[:, j][~np.isnan(XT[:, j])], [10, 50, 90])))
for _, nm, a_, q6, qT in sorted(res, reverse=True):
    print(f"  {nm:22s} AUC {a_:.3f}   day6 p10/50/90 {q6[0]:8.3f} {q6[1]:8.3f} {q6[2]:8.3f}   eval {qT[0]:8.3f} {qT[1]:8.3f} {qT[2]:8.3f}")
import lightgbm as lgb
rng = np.random.default_rng(0)
evid = np.r_[np.repeat(np.arange(len(ev6)), 10), len(ev6) + np.repeat(np.arange(len(hlT)), 10)]
fold = rng.integers(0, 2, evid.max() + 1)[evid]
m = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "verbose": -1, "min_data_in_leaf": 200},
              lgb.Dataset(X[fold == 0], y[fold == 0], weight=w[fold == 0], feature_name=names), 300)
p = m.predict(X[fold == 1])
print(f"\nclassifier AUC (held-out half, weighted): {wauc(p, y[fold == 1], w[fold == 1]):.4f}")
g = sorted(zip(m.feature_importance('gain'), names), reverse=True)[:12]
print("top features by gain: " + ", ".join(f"{n} {v / sum(x for x, _ in g):.2f}" for v, n in g))
