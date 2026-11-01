# Run from the repository root: .venv/bin/python experiments/lab/early_days.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
#
# Why: LightGBM trains on fit days 2-6 only (FIRST_MODEL_DAY = 2), because day 0 has no earlier
# day to train an MF on and day 1 only has day 0. So 196,000 events -- 28% more training rows --
# are dropped entirely rather than included with the MF features marked MISSING, which LightGBM
# handles natively. The base features (repeat clicks, similarity, item/user rates, time) are
# perfectly good on those days.
# Day 6 is split by event: half A for early stopping, AUC reported on half B (baseline 0.83571).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
auc = e13.roc_auc
row_day = np.repeat(fit["day"], 10)
EV = e13.EV_SIZE

def build(day_from):
    """Feature matrix for every row of fit days >= day_from; MF columns are NaN where the MF has
    no score (days < FIRST_MODEL_DAY). Mirrors e13.feature_matrix column for column."""
    ev_idx = np.flatnonzero(fit["day"] >= day_from)
    full = (ev_idx[:, None] * EV + np.arange(EV)).ravel()
    base = np.concatenate([fit["cand"][full], np.repeat(fit["ev"][ev_idx], EV, axis=0)], 1)
    m = mf[full]
    rel = np.full((len(full), 4), np.nan, np.float32)
    ok = ~np.isnan(m[:, 0])
    ok_ev = ok.reshape(-1, EV).all(1)
    if ok_ev.any():                                   # event-relative MF features where available
        sub = m.reshape(-1, EV, m.shape[1])[ok_ev].reshape(-1, m.shape[1])
        rel[np.repeat(ok_ev, EV)] = e13.relative_mf(sub)
    X = np.concatenate([base, m, rel, e13.relative_base(base)], 1)
    names = e13.BASE_FEATURES + e13.MF_FEATURES + e13.REL_FEATURES
    assert X.shape[1] == len(names), (X.shape, len(names))
    return full, X, names

rows, X, names = build(0)
y, d, ev = fit["label"][rows], row_day[rows], rows // EV
print(f"rows: days 0-6 {len(rows):,}; days 2-6 {(d >= 2).sum():,} "
      f"(+{100 * (d < 2).sum() / (d >= 2).sum():.0f}% more)")
print(f"MF features missing on {np.isnan(X[:, len(e13.BASE_FEATURES)]).mean():.1%} of rows "
      f"(days 0-1)", flush=True)

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
A, B = (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

import lightgbm as lgb
def run(label, trm):
    m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(X[trm], y[trm], feature_name=names), 3000,
                  valid_sets=[lgb.Dataset(X[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pb = m.predict(X[B], num_iteration=m.best_iteration)
    print(f"  {label:34s} rows {trm.sum():>9,d}  half-A {m.best_score['valid_0']['auc']:.5f}  "
          f"half-B {auc(pb, y[B]):.5f}  ({m.best_iteration} trees)", flush=True)
print()
run("days 2-5 (= 13)", (d >= 2) & (d <= 5))
run("days 0-5, MF missing on 0-1", d <= 5)
run("days 1-5, MF missing on 1", (d >= 1) & (d <= 5))
