# Run from the repository root: .venv/bin/python experiments/lab/lgb_settings.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# LightGBM settings on 13's features (last tuned on 06's features). Fit days 2-5; day 6 split by
# event: half A for early stopping, AUC reported on half B (baseline 0.83571).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

import lightgbm as lgb
dtr = lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=False)
dA = lgb.Dataset(X[A], y[A], reference=dtr)
variants = {
    "13 as is": {},
    "extra_trees": dict(extra_trees=True),
    "15 leaves, min_data 1000": dict(num_leaves=15, min_data_in_leaf=1000),
    "255 leaves, min_data 500": dict(num_leaves=255, min_data_in_leaf=500),
    "lambda_l2 30, min_gain 1": dict(lambda_l2=30.0, min_gain_to_split=1.0),
    "lr 0.02": dict(learning_rate=0.02),
}
for name, extra in variants.items():
    p = {**e13.LGB_PARAMS, **extra}
    m = lgb.train(p, dtr, 6000, valid_sets=[dA], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(X[B], num_iteration=m.best_iteration)
    print(f"  {name:28s} A {m.best_score['valid_0']['auc']:.5f}  B {e13.roc_auc(pb, y[B]):.5f}  "
          f"({m.best_iteration} trees)", flush=True)
