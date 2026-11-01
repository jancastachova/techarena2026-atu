# Run from the repository root (same arguments as lab/extra_member.py, --base-extra only):
#   .venv/bin/python experiments/lab/light_model.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# 61% of evaluation events have <= 20 known clicks, against 28% of the fit events, so LightGBM learns
# mostly from heavy users. Here a second LightGBM is trained on light users' events only (<= 20
# known clicks; trees by early stopping on half A's light events) and, for light users' rows,
# its probability is averaged with the full model's (weight a). Both are probabilities of the same
# event, so heavy and light rows stay comparable in the global ranking. Reported on half B:
# plain, evaluation-weighted, 1-20 known clicks.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
def fit_pred(mask_tr, mask_a):
    m = lgb.train(p, lgb.Dataset(base_X[mask_tr], y[mask_tr], feature_name=base_names), 8000,
                  valid_sets=[lgb.Dataset(base_X[mask_a], y[mask_a])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    return m.predict(base_X[B], num_iteration=m.best_iteration), m.best_iteration
lightrow = hl_row <= 20
full, nf = fit_pred(tr, A)
lite, nl = fit_pred(tr & lightrow, A & lightrow)
hb = hl_row[B]; light = (hb >= 1) & (hb <= 20); lb = lightrow[B]
print(f"full model {nf} trees, light-only model {nl} trees "
      f"(trained on {np.mean(lightrow[tr]):.1%} of the training rows)")
for a_ in [0.0, 0.3, 0.5, 0.7, 1.0]:
    s = np.where(lb, (1 - a_) * full + a_ * lite, full)
    print(f"  light rows: {1 - a_:.1f} full + {a_:.1f} light-only: half-B {auc(s, y[B]):.5f}  "
          f"eval-weighted {wauc(s, y[B], W_EVAL[B]):.5f}  1-20 known clicks {auc(s[light], y[B][light]):.5f}", flush=True)
