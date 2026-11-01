# Run from the repository root: .venv/bin/python experiments/lab/window_impressions.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
#
# Why: every item-popularity feature is frozen at the cutoff, because click rates are labels.
# But the candidate sets themselves are NOT labels: evaluation.csv gives all 529,390 candidate
# rows up front, so counting how often each item appears there is a label-free measurement of
# item popularity AT EVALUATION TIME, after the frozen statistics have gone stale. RESULTS.md
# dismissed this as "would add little" on the grounds that impression shares are stable (0.93
# between halves of the week) -- a judgement, never a measurement. Here it is measured: for each
# fit day, impressions counted WITHIN that day (the analogue of counting within the evaluation
# window), used as features.
# Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B (baseline 0.83571).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
auc = e13.roc_auc
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))

# how stable are impression counts, and do they drift? (before day 6 vs day 6 itself)
past = np.bincount(fit["item"][row_day < 6], minlength=n_items).astype(np.float64)
now = np.bincount(fit["item"][row_day == 6], minlength=n_items).astype(np.float64)
sh_p, sh_n = past / past.sum(), now / now.sum()
busy = past >= 20
print(f"items with >=20 impressions before day 6: {busy.sum():,}")
print(f"  correlation of impression shares, before day 6 vs day 6: {np.corrcoef(sh_p[busy], sh_n[busy])[0,1]:.4f}")
drift = sh_n[busy] / np.maximum(sh_p[busy], 1e-12)
print(f"  share ratio day6/before: median {np.median(drift):.2f}, 10th pct {np.quantile(drift,0.1):.2f}, "
      f"90th pct {np.quantile(drift,0.9):.2f}")
# does a rising impression share predict clicks the frozen click rate misses?
clk = np.bincount(fit["item"][row_day < 6], weights=fit["label"][row_day < 6].astype(float),
                  minlength=n_items)
ctr_past = (clk + 50 * e13.CTR_PRIOR) / (past + 50)
d6 = row_day == 6
it6 = fit["item"][d6]
ok = busy[it6]
resid = fit["label"][d6].astype(float)[ok] - ctr_past[it6[ok]]
log_drift = np.log(np.maximum(sh_n[it6[ok]], 1e-12)) - np.log(np.maximum(sh_p[it6[ok]], 1e-12))
print(f"  correlation of log(share change) with the day-6 click residual: "
      f"{np.corrcoef(log_drift, resid)[0, 1]:+.4f}", flush=True)

# features: impressions of the candidate WITHIN the scored day, its share, and the change
# against the frozen pre-day count
NEW = np.zeros((len(rows), 3), np.float32)
for day in range(2, 7):
    sel = np.flatnonzero(row_day == day)
    cnt_in = np.bincount(fit["item"][sel], minlength=n_items).astype(np.float64)
    cnt_before = np.bincount(fit["item"][row_day < day], minlength=n_items).astype(np.float64)
    sh_in = cnt_in / cnt_in.sum()
    sh_bf = cnt_before / max(cnt_before.sum(), 1)
    it = fit["item"][sel]
    NEW[pos[sel], 0] = np.log1p(cnt_in[it])
    NEW[pos[sel], 1] = np.log(np.maximum(sh_in[it], 1e-12))
    NEW[pos[sel], 2] = np.log(np.maximum(sh_in[it], 1e-12)) - np.log(np.maximum(sh_bf[it], 1e-12))

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

import lightgbm as lgb
def run(label, Xm, nm):
    m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 3000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    print(f"  {label:36s} half-A {m.best_score['valid_0']['auc']:.5f}  half-B {auc(pb, y[B]):.5f}  "
          f"({m.best_iteration} trees)", flush=True)
    return m
print()
run("13 as is", X, names)
nm = ["log_impr_in_window", "log_share_in_window", "log_share_change"]
m = run("+ in-window impressions", np.concatenate([X, NEW], 1), names + nm)
gains = sorted(zip(m.feature_importance("gain"), names + nm), reverse=True)
print("  rank of the new features by gain: " +
      ", ".join(f"{n} #{i+1}" for i, (g, n) in enumerate(gains) if n in nm))
