# Run from the repository root (same arguments as lab/extra_member.py, --base-extra only):
#   .venv/bin/python experiments/lab/decompose.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# Where is the remaining global AUC lost: ordering inside events, or ranking events against each
# other? Trains LightGBM (lr 0.02) on the current best feature set (fit days 2-5, early stopping
# on half A of day 6) and splits its half-B predictions:
#   within    mean per-event AUC (events with a click)
#   event     AUC of "the event gets a click" from the predicted click probabilities summed
#   oracle    global AUC when one part is replaced by the truth: true event outcome with our
#             within-event order, and our event level with a perfect within-event order
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
# reuse extra_member.py up to the LightGBM part (features, halves)
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
m = lgb.train(p, lgb.Dataset(base_X[tr], y[tr], feature_name=base_names), 8000,
              valid_sets=[lgb.Dataset(base_X[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
pb = m.predict(base_X[B], num_iteration=m.best_iteration)
yb = y[B].reshape(-1, 10); P = pb.reshape(-1, 10)
assert np.all(ev[B].reshape(-1, 10) == ev[B].reshape(-1, 10)[:, :1])
print(f"half-B global AUC {auc(pb, y[B]):.5f} ({len(P):,} events, {yb.max(1).mean():.3f} with a click)")
has = yb.max(1) > 0
w = np.mean([auc(P[i], yb[i]) for i in np.flatnonzero(has)])
print(f"within-event AUC (mean over clicked events) {w:.5f}")
s = P.sum(1)
print(f"event-level AUC (sum of p vs has click) {auc(s, has.astype(int)):.5f}; max p {auc(P.max(1), has.astype(int)):.5f}")
# oracle 1: true event outcome, our within-event order -> score = has*10 + within-event share
share = P / P.sum(1, keepdims=True)
o1 = (has[:, None] * 10 + share).ravel()
print(f"oracle event level, our within order: {auc(o1, yb.ravel()):.5f}")
# oracle 2: our event level (sum p), perfect within order: clicked candidate gets the event's
# total, the others share it below
o2 = np.where(yb > 0, s[:, None], s[:, None] * 0.5 * share).ravel()
print(f"our event level, perfect within order: {auc(o2, yb.ravel()):.5f}")
# by user history length
hl = np.repeat(fit["hlen"], 10)[rows][B].reshape(-1, 10)[:, 0] if "hlen" in fit else None
hist_n = hlen[ev[B].reshape(-1, 10)[:, 0]]
for lo, hi in [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]:
    k = (hist_n >= lo) & (hist_n <= hi)
    kk = np.repeat(k, 10)
    wi = np.mean([auc(P[i], yb[i]) for i in np.flatnonzero(k & has)])
    print(f"  known clicks {lo:3d}-{hi:3d}: {k.mean():6.1%} of events, global AUC {auc(pb[kk], y[B][kk]):.5f}, "
          f"within {wi:.5f}, event-level {auc(s[k], has[k].astype(int)):.5f}, click share {has[k].mean():.3f}")
