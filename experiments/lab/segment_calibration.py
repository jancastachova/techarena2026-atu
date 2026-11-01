# Run from the repository root: .venv/bin/python experiments/lab/segment_calibration.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
#
# Why: the metric is GLOBAL AUC over all rows, so how score LEVELS line up BETWEEN kinds of rows
# matters, not only the order within them. The segments differ a lot (first-event click rates:
# users with no known clicks 4.5%, 21+ clicks 6.6%), so if the model's scores are mutually
# misaligned across segments, a monotone per-segment transform raises global AUC without changing
# any within-segment order. Needs no retraining: the transform is fitted on half A of day 6 and
# read on half B (baseline 0.83571).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
auc = e13.roc_auc
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
known = np.repeat(hlen.astype(np.int64), 10)[rows]

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

import lightgbm as lgb
m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=names), 3000,
              valid_sets=[lgb.Dataset(X[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
pA, pB = m.predict(X[A], num_iteration=m.best_iteration), m.predict(X[B], num_iteration=m.best_iteration)
base = auc(pB, y[B])
print(f"13 as is: half-B {base:.5f} ({m.best_iteration} trees)\n", flush=True)

seg_of = lambda k: np.where(k == 0, 0, np.where(k <= 5, 1, np.where(k <= 20, 2, 3)))
sA, sB = seg_of(known[A]), seg_of(known[B])
names_seg = ["0 clicks", "1-5", "6-20", "21+"]
print(f"  {'segment':10s} {'rows A':>8s} {'click rate':>11s} {'mean score':>11s} {'ratio':>7s}")
for i, nm in enumerate(names_seg):
    mA = sA == i
    if mA.sum():
        print(f"  {nm:10s} {mA.sum():8,d} {y[A][mA].mean():11.4f} {pA[mA].mean():11.4f} "
              f"{pA[mA].mean() / max(y[A][mA].mean(), 1e-9):7.3f}")

def calibrate(p_fit, y_fit, s_fit, p_new, s_new, n_bins):
    """Per segment: replace the score by the empirical click rate of its quantile bin."""
    out = p_new.copy().astype(np.float64)
    for i in range(len(names_seg)):
        f, n = s_fit == i, s_new == i
        if f.sum() < 2000 or not n.sum():
            continue
        edges = np.quantile(p_fit[f], np.linspace(0, 1, n_bins + 1)[1:-1])
        bf = np.searchsorted(edges, p_fit[f])
        bn = np.searchsorted(edges, p_new[n])
        cnt = np.bincount(bf, minlength=n_bins).astype(np.float64)
        clk = np.bincount(bf, weights=y_fit[f].astype(np.float64), minlength=n_bins)
        prior = y_fit[f].mean()
        rate = (clk + 20 * prior) / (cnt + 20)            # shrunk to the segment's own rate
        out[n] = rate[bn]
    return out

print()
for n_bins in (10, 25, 50, 100):
    cB = calibrate(pA, y[A], sA, pB, sB, n_bins)
    cA = calibrate(pA, y[A], sA, pA, sA, n_bins)          # in-sample, for reference only
    print(f"per-segment calibration, {n_bins:3d} bins: half-B {auc(cB, y[B]):.5f}  "
          f"gain {auc(cB, y[B]) - base:+.5f}   (in-sample half-A {auc(cA, y[A]):.5f})", flush=True)

# a single global shift per segment (one number each), the mildest possible version
print()
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
shift = np.zeros(len(names_seg))
for i in range(len(names_seg)):
    f = sA == i
    if f.sum() > 2000:
        shift[i] = lg(max(y[A][f].mean(), 1e-6)) - lg(max(pA[f].mean(), 1e-6))
oB = lg(pB) + shift[sB]
print(f"per-segment log-odds shift only: half-B {auc(oB, y[B]):.5f}  gain {auc(oB, y[B]) - base:+.5f}")
print("  shifts: " + ", ".join(f"{nm} {shift[i]:+.3f}" for i, nm in enumerate(names_seg)))
