# Run from the repository root: .venv/bin/python experiments/lab/oof_check.py [data/cache/28_oofval_k5_s3.npz]
# Needs the validation file of experiments/28_oof_mf/oof_mf.py --cutoff 6.
#
# Out-of-fold vs time-forward MF features for LightGBM, with day 6 as the test day (scored in all
# variants by the MF trained on days 0-5, as the evaluation rows are by the 7-day MF):
#   A  time-forward scores (current scheme), training days 2-5
#   B  out-of-fold scores, training days 2-5      (stronger features, same rows)
#   C  out-of-fold scores, training days 0-5      (+ the two days the current scheme drops)
# Base features + one MF (13's feature_matrix: MF scores + event-relative versions), LightGBM lr
# 0.02, early stopping on half A of day 6, AUC on half B: plain and evaluation-weighted (events
# reweighted to evaluation.csv's mix of known clicks, see lab/extra_member.py).
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/cache/28_oofval_k5_s3.npz"))
auc = e13.roc_auc
row_day = np.repeat(fit["day"], 10)
d6 = row_day == 6
def with_day6(m):
    m = m.copy(); m[d6] = z["day6"][d6]; return m
TF, OOF = with_day6(z["tf"]), with_day6(z["oof"])
import os
NOBU = os.environ.get("NOBU") == "1"     # (2026-11-02) by-user folds leave the scored user's MF bias untrained (0), so
def nobu(m):                             # every variant drops it: logit - b_user, b_user column 0 (base features keep
    m = m.copy(); m[:, 0] -= m[:, 2]; m[:, 2] = 0; return m   # the user click rates)

_, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
_, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def bucket(h):
    b = np.zeros(len(h), int)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(h >= lo) & (h <= hi)] = k
    return b
ev6_all = np.flatnonzero(fit["day"] == 6)
share6 = np.bincount(bucket(hlen[ev6_all].astype(int)), minlength=5) / len(ev6_all)
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / share6
def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]
    g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))

rng = np.random.default_rng(0)
ev6 = np.unique(np.flatnonzero(row_day >= 2)[d6[row_day >= 2]] // 10)
a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True

import lightgbm as lgb
saved = {}
def run(label, mf, first_day):
    rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= first_day))
    y, d, ev = fit["label"][rows], row_day[rows], rows // 10
    tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=names), 8000,
                  valid_sets=[lgb.Dataset(X[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(X[B], num_iteration=m.best_iteration)
    w = ratio[bucket(hlen[ev[B]].astype(int))]
    saved[label] = (rows[B], pb)
    print(f"  {label:44s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], w):.5f}  "
          f"({tr.sum() // 10:,} training events, {m.best_iteration} trees)", flush=True)
r6 = d6
print(f"MF alone on day 6: {auc(z['day6'][r6, 0], fit['label'][r6]):.5f}")
run("A time-forward, days 2-5 (current)", TF, 2)
if NOBU:
    run("A0 time-forward without b_user", nobu(TF), 2)
    OOF = nobu(OOF)
run("B out-of-fold, days 2-5", OOF, 2)
run("C out-of-fold, days 0-5", OOF, 0)
ra = saved["A time-forward, days 2-5 (current)"][0]
assert all(np.array_equal(v[0], ra) for v in saved.values())
rows2 = np.flatnonzero(row_day >= 2)                  # lab/paired_ci.py indexes the rows of days >= 2
np.savez(str(ROOT / os.environ.get("OUT", "data/cache/lab_preds_oof_check.npz")), rows_B=np.searchsorted(rows2, ra),
         **{f"{k}|B": v[1] for k, v in saved.items()})
