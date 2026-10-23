# Run from the repository root: .venv/bin/python experiments/lab/combined_small_gains.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Do the three noise-level gains of 2026-10-23 add up? popularity-taste features, LightGBM
# learning rate 0.02, XGBoost rank blend. Fit days 2-5; day 6 halves: A = early stopping and
# blend weight, B = report (baseline 0.83571).
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
auc = e13.roc_auc
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
y, d, ev = fit["label"][rows], row_day[rows], rows // 10

POP = np.zeros((len(rows), 2), np.float32)                       # as-of popularity taste, per fit day
pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))
for day in range(2, 7):
    keep = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
    lp = np.log1p(np.bincount(fit["clk_item"][keep], minlength=n_items)).astype(np.float32)
    e = np.flatnonzero(fit["day"] == day)
    hl = hlen[e].astype(np.float32)
    m = np.arange(hist.shape[1])[None, :] < hl[:, None]
    hp = np.where(hl > 0, np.where(m, lp[hist[e]], 0).sum(1) / np.maximum(hl, 1), np.nan)
    cp = lp[fit["item"].reshape(-1, 10)[e]]
    POP[pos[(e[:, None] * 10 + np.arange(10)).ravel()]] = np.stack([np.repeat(hp, 10), cp.ravel() - np.repeat(hp, 10)], 1)
Xp, names_p = np.concatenate([X, POP], 1), names + ["hist_pop", "cand_pop_minus_hist"]

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

import lightgbm as lgb, xgboost as xgb
def lgbm(Xm, nm, lr):
    p = {**e13.LGB_PARAMS, "learning_rate": lr}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    return m.predict(Xm[A], num_iteration=m.best_iteration), m.predict(Xm[B], num_iteration=m.best_iteration)
def xgbm(Xm):
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", eta=0.05, max_depth=6,
             min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, seed=0, nthread=0)
    dA = xgb.DMatrix(Xm[A], y[A])
    m = xgb.train(p, xgb.DMatrix(Xm[tr], y[tr]), 4000, evals=[(dA, "A")], early_stopping_rounds=100, verbose_eval=False)
    it = (0, m.best_iteration + 1)
    return m.predict(dA, iteration_range=it), m.predict(xgb.DMatrix(Xm[B]), iteration_range=it)
def blend(a, b):
    r = lambda p: rankdata(p) / len(p)
    ws = np.linspace(0, 1, 11)
    w = ws[int(np.argmax([auc((1 - w) * r(a[0]) + w * r(b[0]), y[A]) for w in ws]))]
    return w, auc((1 - w) * r(a[1]) + w * r(b[1]), y[B])

base = lgbm(X, names, 0.05)
print(f"13 as is                               B {auc(base[1], y[B]):.5f}", flush=True)
l2 = lgbm(Xp, names_p, 0.02)
print(f"+ popularity taste, lr 0.02            B {auc(l2[1], y[B]):.5f}", flush=True)
x = xgbm(Xp)
print(f"XGBoost with popularity taste          B {auc(x[1], y[B]):.5f}", flush=True)
w, b = blend(l2, x)
print(f"all three (XGBoost weight {w:.1f} from A)   B {b:.5f}   gain {b - auc(base[1], y[B]):+.5f}", flush=True)
