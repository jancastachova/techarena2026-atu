# Run from the repository root: .venv/bin/python experiments/lab/xgb_blend.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Fit-period check of a LightGBM + XGBoost blend (official split, experiment-13 features).
# Train on fit days 2-5; day 6 split by event into halves A/B. Both models early-stop on A,
# blend weight is chosen on A, everything is reported on B (untouched).
import glob, importlib.util, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); half_a = set(rng.choice(ev6, len(ev6) // 2, replace=False).tolist())
in_a = np.array([e in half_a for e in ev])
tr, A, B = d <= 5, (d == 6) & in_a, (d == 6) & ~in_a
auc = e13.roc_auc
print(f"rows train {tr.sum():,}  A {A.sum():,}  B {B.sum():,}", flush=True)

import lightgbm as lgb
t = time.time()
ml = lgb.train(e13.LGB_PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=names), 3000,
               valid_sets=[lgb.Dataset(X[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
pl_A, pl_B = ml.predict(X[A], num_iteration=ml.best_iteration), ml.predict(X[B], num_iteration=ml.best_iteration)
print(f"LightGBM  A {auc(pl_A, y[A]):.5f}  B {auc(pl_B, y[B]):.5f}  ({ml.best_iteration} trees, {time.time()-t:.0f}s)", flush=True)

import xgboost as xgb
dtr, dA, dB = xgb.DMatrix(X[tr], y[tr]), xgb.DMatrix(X[A], y[A]), xgb.DMatrix(X[B])
configs = {
    "xgb depth6": dict(max_depth=6, min_child_weight=50),
    "xgb depth8": dict(max_depth=8, min_child_weight=100),
}
best = None
for name, extra in configs.items():
    t = time.time()
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", eta=0.05,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, seed=0, nthread=0, **extra)
    mx = xgb.train(p, dtr, 3000, evals=[(dA, "A")], early_stopping_rounds=100, verbose_eval=False)
    it = (0, mx.best_iteration + 1)
    px_A, px_B = mx.predict(dA, iteration_range=it), mx.predict(dB, iteration_range=it)
    print(f"{name:9s} A {auc(px_A, y[A]):.5f}  B {auc(px_B, y[B]):.5f}  ({it[1]} trees, {time.time()-t:.0f}s)", flush=True)
    if best is None or auc(px_A, y[A]) > best[0]:
        best = (auc(px_A, y[A]), name, px_A, px_B)

_, name, px_A, px_B = best
r = lambda p: rankdata(p) / len(p)
ws = np.linspace(0, 1, 11)
blend_A = [auc((1 - w) * r(pl_A) + w * r(px_A), y[A]) for w in ws]
w = ws[int(np.argmax(blend_A))]
print(f"\nblend with {name} (rank average)")
for wi, a in zip(ws, blend_A):
    print(f"  w_xgb {wi:.1f}  A {a:.5f}  B {auc((1 - wi) * r(pl_B) + wi * r(px_B), y[B]):.5f}")
print(f"chosen on A: w_xgb {w:.1f} -> B {auc((1 - w) * r(pl_B) + w * r(px_B), y[B]):.5f}  "
      f"vs LightGBM alone B {auc(pl_B, y[B]):.5f}")
