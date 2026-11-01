# Run from the repository root (same arguments as lab/extra_member.py, --base-extra = the full set):
#   .venv/bin/python experiments/lab/first_of_day.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# evaluation.csv holds each user's FIRST event after a time boundary; day 6 half B holds all events,
# including later events of a day that follow a session the model sees. The fit-period analogue of
# an evaluation event is each user's first event of a day. Tested on the 26_ensemble_bundle set:
#   features  position of the event in the user's day (0 = first) and log hours since the user's
#             previous event of any kind (the base features count from the last click)
#   weights   first-of-day events weighted x2 / x4 in training
# Reported on half B (plain, evaluation-weighted) and on half B's first-of-day events alone.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
ts, us, dd = fit["ts"].astype(np.int64), fit["user"].astype(np.int64), fit["day"].astype(np.int64)
o = np.lexsort((ts, us))
pos_day = np.zeros(len(ts), int); gap = np.full(len(ts), np.nan)
same_u = np.r_[False, us[o][1:] == us[o][:-1]]
same_ud = same_u & np.r_[False, dd[o][1:] == dd[o][:-1]]
cnt = 0
for k, e in enumerate(o):
    cnt = cnt + 1 if same_ud[k] else 0
    pos_day[e] = cnt
gap[o[1:][same_u[1:]]] = (ts[o][1:][same_u[1:]] - ts[o][:-1][same_u[1:]]) / 3600
F = np.stack([pos_day[ev], np.log1p(gap[ev])], 1).astype(np.float32)
first = pos_day[ev] == 0
print(f"first-of-day events: {first[d == 6].mean():.1%} of day-6 rows, {first[tr].mean():.1%} of training rows", flush=True)
P = {**e13.LGB_PARAMS, "learning_rate": 0.02}
def run(label, Xm, nm, w=None):
    m = lgb.train(P, lgb.Dataset(Xm[tr], y[tr], weight=w, feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    fb = first[B]
    print(f"  {label:36s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  "
          f"first-of-day only {auc(pb[fb], y[B][fb]):.5f} (eval-weighted {wauc(pb[fb], y[B][fb], W_EVAL[B][fb]):.5f})  "
          f"({m.best_iteration} trees)", flush=True)
run("current", base_X, base_names)
run("+ position in day, hours since prev event", np.concatenate([base_X, F], 1), base_names + ["pos_in_day", "log_h_since_prev_event"])
for k in [2.0, 4.0]:
    run(f"first-of-day weighted x{k:g}", base_X, base_names, w=np.where(first[tr], k, 1.0))
