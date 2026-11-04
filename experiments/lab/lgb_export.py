# Run from the repository root:
#   .venv/bin/python experiments/lab/lgb_export.py data/cache/n1/exp45_d6.npz --add <cache.npz> ... --out <preds.npz>
# 2026-11-03: the FAST path for LightGBM feature screens. lab/extra_member.py rebuilds the whole feature matrix from ~20
# caches on one core (30-40 min on the EPYC box) before every fit; the EXPORT of the same matrix (EXPORT=... extra_member.py)
# already holds it with the split (y, A, B, weights). Here: load the export, append the --add caches (mf_fit columns, rows
# = 10 x event + position, as lab/choice_model.py), train the lab LightGBM (13's params, lr 0.02, early stopping 150 on half
# A, fit days 2..T-1) and save half-B predictions in the lab/paired_ci.py format. --lgbp '{"num_leaves": 15}' overrides.
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("export")
ap.add_argument("--add", action="append", default=[])
ap.add_argument("--out", required=True)
ap.add_argument("--lgbp", default="{}")
ap.add_argument("--rank", type=int, default=0, help="lambdarank over random groups of this many events, fixed --trees "
                "(2026-11-03: the 48 member's settings, G500 / 2200 trees at lr 0.02)")
ap.add_argument("--trees", type=int, default=2200)
ap.add_argument("--threads", type=int, default=int(os.environ.get("NTHREADS", 16)))
a = ap.parse_args()
import lightgbm as lgb
Z = np.load(a.export)
X, y, A, B = Z["X"], Z["y"].astype(np.float32), Z["A"], Z["B"]
names = list(Z["names"])
if a.add:
    rows = Z["ev"] * 10 + np.arange(len(Z["ev"])) % 10
    assert np.all(rows // 10 == Z["ev"])
    for f in a.add:
        M = np.load(f)["mf_fit"][rows].astype(np.float32)
        X = np.concatenate([X, M], 1)
        names += [f"{os.path.basename(f).split('.')[0]}_{j}" for j in range(M.shape[1])]
tr = ~(A | B) & (Z["d"] < Z["d"][A].min())
W_EVAL, FIRST, W_FIRST = Z["W_EVAL"], Z["FIRST"], Z["W_FIRST"]
P = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True, force_row_wise=True,
         num_threads=a.threads)
P.update(json.loads(a.lgbp))
fn = [n.replace(" ", "_") for n in names]
if a.rank:
    ti = np.flatnonzero(tr); ue = np.unique(Z["ev"][ti]); np.random.default_rng(0).shuffle(ue)
    ro = np.empty(ue.max() + 1, np.int64); ro[ue] = np.arange(len(ue))
    key = ro[Z["ev"][ti]]; o = np.argsort(key, kind="stable"); ti = ti[o]
    P.update(objective="lambdarank", lambdarank_truncation_level=10000, label_gain=[0, 1])
    m = lgb.train(P, lgb.Dataset(X[ti], y[ti], group=np.bincount(key[o] // a.rank), feature_name=fn), a.trees)
    m.best_iteration = a.trees
else:
    m = lgb.train(P, lgb.Dataset(X[tr], y[tr], feature_name=fn), 8000,
                  valid_sets=[lgb.Dataset(X[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
pb = m.predict(X[B], num_iteration=m.best_iteration)


def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


yb, fb = y[B], FIRST[B]
print(f"  lgb_export {os.path.basename(a.export)} +{len(a.add)} caches: half-B {wauc(pb, yb, np.ones(len(yb))):.5f}  "
      f"eval-weighted {wauc(pb, yb, W_EVAL[B]):.5f}  first {wauc(pb[fb], yb[fb], W_FIRST[B][fb]):.5f}  "
      f"({m.best_iteration} trees)", flush=True)
np.savez(a.out, rows_B=np.flatnonzero(B), **{"base (= the best member)|B": pb})
print("wrote", a.out)
