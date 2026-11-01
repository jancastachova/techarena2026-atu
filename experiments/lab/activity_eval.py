# Run from the repository root (same arguments as lab/extra_member.py, --base-extra only):
#   .venv/bin/python experiments/lab/activity_eval.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# lab/user_activity.py's user-activity features (click counts over windows, click rate, events in
# train, clicks per event) were noise on half B (0.83842 -> 0.83849). They describe light and
# occasional users, whom evaluation.csv weighs ~4x more than day 6, and the other team's file
# leans on them. Rescreened here on the 26_ensemble_hops feature set with the evaluation-weighted
# half-B AUC of lab/extra_member.py.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
ua = (ROOT / "experiments/lab/user_activity.py").read_text()
exec(ua[ua.index("def activity"):ua.index("ACT = activity")])
ACT = activity(fit, fit)
SETS = {"all": list(ACT), "events": ["act_log_events", "act_clicks_per_event"],
        "rate": ["act_rate20_per_day", "act_mean_gap_h", "act_log_span_h"]}
import lightgbm as lgb
def run(label, Xm, nm):
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    hb = hl_row[B]; light = (hb >= 1) & (hb <= 20)
    print(f"  {label:32s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  "
          f"1-20 known clicks {auc(pb[light], y[B][light]):.5f}  ({m.best_iteration} trees)", flush=True)
run("current best", base_X, base_names)
for s_, cols in SETS.items():
    F = np.stack([ACT[c][rows // 10] for c in cols], 1)
    run(f"+ activity ({s_})", np.concatenate([base_X, F], 1), base_names + cols)
