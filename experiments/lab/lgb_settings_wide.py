# Run from the repository root (same arguments as lab/extra_member.py, --base-extra only):
#   .venv/bin/python experiments/lab/lgb_settings_wide.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# LightGBM settings were last tuned on 13's ~60 features (lab/lgb_settings.py); the current best
# set (26_ensemble_hops) has ~200, most of them correlated views of the same scores. Narrower
# column sampling and other leaf sizes may now fit better. lr 0.02 throughout (the current
# member). Fit days 2-5; day 6 half A early stopping, half B AUC.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
print(f"{base_X.shape[1]} features", flush=True)
base = {**e13.LGB_PARAMS, "learning_rate": 0.02}
settings = [
    ("current (63 leaves, ff 0.8, min leaf 200)", {}),
    ("feature_fraction 0.5", {"feature_fraction": 0.5}),
    ("feature_fraction 0.3", {"feature_fraction": 0.3}),
    ("127 leaves", {"num_leaves": 127}),
    ("31 leaves", {"num_leaves": 31}),
    ("min leaf 1000", {"min_data_in_leaf": 1000}),
    ("lambda_l2 10", {"lambda_l2": 10.0}),
    ("ff 0.5 + 127 leaves + min leaf 500", {"feature_fraction": 0.5, "num_leaves": 127, "min_data_in_leaf": 500}),
]
for label, extra in settings:
    m = lgb.train({**base, **extra}, lgb.Dataset(base_X[tr], y[tr], feature_name=base_names), 10000,
                  valid_sets=[lgb.Dataset(base_X[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(base_X[B], num_iteration=m.best_iteration)
    print(f"  {label:44s} half-B {auc(pb, y[B]):.5f}  half-A {m.best_score['valid_0']['auc']:.5f}  "
          f"({m.best_iteration} trees)", flush=True)
