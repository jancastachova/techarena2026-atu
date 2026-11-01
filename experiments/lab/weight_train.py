# Run from the repository root (same arguments as lab/extra_member.py, --base-extra only):
#   .venv/bin/python experiments/lab/weight_train.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# evaluation.csv weighs light users ~4x more than the fit events (RESULTS.md). LightGBM is trained
# with each row weighted by (evaluation share / fit share) of its known-clicks bucket, to the power
# p (0 = current), shares measured on the training days themselves; trees by early stopping on half
# A (unweighted, as before). Reported: half B, evaluation-weighted half B and 1-20 known clicks.
# Tried once on 13's features (1/user events, no gain); the audience / hop features of experiment
# 26 matter most for light users, so it is retested on the current best set.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
btr = bucket(hl_row[tr][::10])
share_tr = np.bincount(btr, minlength=len(BUCKETS)) / len(btr)
ratio = share_ev / np.maximum(share_tr, 1e-9)
print("training-days bucket ratios: " + ", ".join(f"{lo}-{hi} {r_:.2f}" for (lo, hi), r_ in zip(BUCKETS, ratio)))
for pw in [0.0, 0.5, 1.0]:
    w = ratio[bucket(hl_row[tr])] ** pw
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(base_X[tr], y[tr], weight=w, feature_name=base_names), 8000,
                  valid_sets=[lgb.Dataset(base_X[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(base_X[B], num_iteration=m.best_iteration)
    hb = hl_row[B]; light = (hb >= 1) & (hb <= 20)
    print(f"  weights^{pw:.1f}: half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  "
          f"1-20 known clicks {auc(pb[light], y[B][light]):.5f}  ({m.best_iteration} trees)", flush=True)
