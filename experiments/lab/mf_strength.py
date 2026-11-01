# Run from the repository root (same arguments as lab/extra_member.py, --base-extra only):
#   .venv/bin/python experiments/lab/mf_strength.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# LightGBM learns how far to trust the MF scores from rows scored by WEAKER MFs than the test rows
# get: fit day d is scored by an MF trained on d days (MF13 alone: day 2 AUC 0.781 ... day 6 0.819)
# while the score scale stays the same (sd 1.27-1.30), so one score value means more on later days;
# the test rows are scored by a 7-day MF. Variants (trained on days 2-5, scored on day 6, which is
# unseen like the test days):
#   day weights     row weight (day - 1)^p: later days, closer in MF quality to the test, count more
#   day feature     the scoring day as a feature (trees cannot extrapolate: unseen days are treated
#                   like the latest training day, the closest in MF quality)
# Half-B predictions are saved (data/cache/lab_preds_mf_strength.npz) for lab/paired_ci.py.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
saved = {"rows_B": np.flatnonzero(B)}
def run(label, Xm, nm, w=None):
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], weight=w, feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    saved[f"{label}|B"] = pb
    hb = hl_row[B]; light = (hb >= 1) & (hb <= 20)
    print(f"  {label:24s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  "
          f"1-20 known clicks {auc(pb[light], y[B][light]):.5f}  ({m.best_iteration} trees)", flush=True)
run("base", base_X, base_names)
for pw in [1.0, 2.0]:
    run(f"day weights p={pw:g}", base_X, base_names, w=(d[tr] - 1.0) ** pw)
run("day feature", np.concatenate([base_X, d[:, None].astype(np.float32)], 1), base_names + ["score_day"])
np.savez(str(CACHE / "lab_preds_mf_strength.npz"), **saved)
