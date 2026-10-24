# Run from the repository root: .venv/bin/python experiments/lab/seq_feature.py <seq_mf.npz>
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Produce the sequence scores first:
#   .venv/bin/python experiments/18_sequence/sequence.py --cutoff 7 --pure-seq --mf-seeds 1
# (the path it prints as "MF scores:" is the argument here)
#
# Why: the standalone sequence model (raw item ID sequences, no mean-of-clicks anchor) scores
# 0.798 on day 6 against the MF's 0.817, but its predictions correlate only 0.903 with the MF --
# the first genuinely diverse member we have (everything else sits at 0.995+). A weak but
# diverse member is used by WEIGHTING it, not by averaging it equally. Two ways here:
#   1. as extra LightGBM features next to the MF features (trees can use it conditionally);
#   2. as a fitted rank blend against the finished LightGBM predictions.
# Fit days 2-5; day 6 split by event: half A for early stopping and the blend weight, AUC on
# half B (baseline 0.83571).
import importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
SEQ = sys.argv[1] if len(sys.argv) > 1 else None
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
if SEQ is None:
    cands = sorted(ROOT.glob("data/cache/18_mf_c7_*.npz"))
    if not cands:
        sys.exit("no 18_mf_c7_*.npz found: run the pure-sequence scoring first")
    SEQ = str(cands[-1])
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
sq = np.load(SEQ)["mf_fit"]
print(f"sequence scores: {Path(SEQ).name}")
auc = e13.roc_auc
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
S = sq[rows]
if np.isnan(S).any():
    sys.exit("sequence scores have NaNs for scored rows")

# event-relative versions of the sequence logit, as for the MF
srel = e13.relative_mf(S)
snames = ["seq_logit", "seq_item_bias", "seq_user_bias", "seq_dot",
          "seq_minus_max", "seq_rank", "seq_share", "seq_lse"]
Xs = np.concatenate([X, S, srel], 1)

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]
print(f"day-6 AUC alone: MF {auc(mf[rows][d == 6, 0], y[d == 6]):.5f}  "
      f"sequence {auc(S[d == 6, 0], y[d == 6]):.5f}  "
      f"rank corr {np.corrcoef(rankdata(mf[rows][d == 6, 0]), rankdata(S[d == 6, 0]))[0, 1]:.4f}", flush=True)

import lightgbm as lgb
def run(label, Xm, nm):
    m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 3000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pa, pb = m.predict(Xm[A], num_iteration=m.best_iteration), m.predict(Xm[B], num_iteration=m.best_iteration)
    print(f"  {label:34s} half-A {auc(pa, y[A]):.5f}  half-B {auc(pb, y[B]):.5f}  ({m.best_iteration} trees)",
          flush=True)
    return pa, pb, m
print()
pa, pb, _ = run("13 as is", X, names)
base_B = auc(pb, y[B])
_, _, ms = run("+ sequence features", Xs, names + snames)
gains = sorted(zip(ms.feature_importance("gain"), names + snames), reverse=True)
print("  sequence features in the top 15 by gain: " +
      ", ".join(f"{n} ({g:,.0f})" for g, n in gains[:15] if n in snames) or "  none in the top 15")

r = lambda p: rankdata(p) / len(p)
ws = np.linspace(0, 0.5, 11)
best = max((auc((1 - w) * r(pa) + w * r(S[A, 0]), y[A]), w) for w in ws)
w = best[1]
blend_B = auc((1 - w) * r(pb) + w * r(S[B, 0]), y[B])
print(f"\nfitted rank blend of the LightGBM model with the sequence score:")
for wi in ws[::2]:
    print(f"  w_seq {wi:.2f}  half-A {auc((1-wi)*r(pa) + wi*r(S[A,0]), y[A]):.5f}  "
          f"half-B {auc((1-wi)*r(pb) + wi*r(S[B,0]), y[B]):.5f}")
print(f"  chosen on half A: w_seq {w:.2f} -> half B {blend_B:.5f}  gain {blend_B - base_B:+.5f}")
