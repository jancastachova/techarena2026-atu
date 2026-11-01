# Run from the repository root:
#   .venv/bin/python experiments/lab/paired_ci.py <A.npz>[:key] <B.npz>[:key] [--n 300]
# A file is either a members file of lab/ensemble_v5.py --save-members (no key: the equal-weight rank
# average of LGB / XGB / MLP, as in the submission) or a predictions file of lab/extra_member.py
# --save-preds (key = the run's label).
#
# Are two half-B results really different? Single comparisons this session were read at the fifth
# decimal without error bars. Paired bootstrap over half-B EVENTS (all 10 rows of an event move
# together; both models scored on the same resample): mean difference and 95% interval of B - A, for
# the plain AUC and the evaluation-weighted AUC (events reweighted to evaluation.csv's mix of known
# clicks, as in lab/extra_member.py).
import argparse, importlib.util, os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--first", action="store_true",
                help="keep only each user's FIRST day-6 event (the shape of evaluation.csv: one first event per user)")
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
y_all = fit["label"][rows]
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
_, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
hl_row = hlen[rows // 10].astype(int)
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def bucket(h):
    b = np.zeros(len(h), int)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(h >= lo) & (h <= hi)] = k
    return b
TEST_DAY = int(os.environ.get("TEST_DAY", 6))     # as in lab/extra_member.py
d6 = np.repeat(fit["day"], 10)[rows] == TEST_DAY
b6 = bucket(hl_row[d6][::10])
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(b6, minlength=5) / len(b6))

def load(spec_):
    path, _, key = spec_.partition(":")
    z = np.load(path)
    if key:
        return z["rows_B"], z[f"{key}|B"]
    r = lambda p: rankdata(p) / len(p)
    return z["rows_B"], sum(r(z[f"{k}|B"]) for k in ("LGB full (lr .02)", "XGB full", "MLP full"))
ra, pa = load(args.a); rb, pb = load(args.b)
assert np.array_equal(ra, rb), "the two files are not on the same half-B rows"
if args.first:
    # first day-6 event of each user (fit events are in time order); the evaluation weights are recomputed for that subset
    ev_all = rows[ra] // 10
    d6ev = np.flatnonzero(fit["day"] == TEST_DAY)
    _, fi = np.unique(fit["user"][d6ev], return_index=True)
    first = np.zeros(len(fit["day"]), bool); first[d6ev[fi]] = True
    keep = first[ev_all]
    ra, pa, pb = ra[keep], pa[keep], pb[keep]
    fb = bucket(hlen[d6ev[fi]].astype(int))
    ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(fb, minlength=5) / len(fb))
    print(f"--first: {keep.sum() // 10:,} of {len(keep) // 10:,} half-B events are users' first events of day {TEST_DAY}")
y = y_all[ra]; w_eval = ratio[bucket(hl_row[ra])]
n_ev = len(y) // 10

def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]
    g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))

# gap-weighted (2026-10-31): events reweighted to evaluation's hours-since-last-click mix (see lab/extra_member.py, W_GAP)
cu, ct = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64); o_ = np.lexsort((ct, cu)); cu, ct = cu[o_], ct[o_]
def gap_bucket(src):
    key = src["user"].astype(np.int64) * 10**10 + src["ts"].astype(np.int64)
    j = np.searchsorted(cu * 10**10 + ct, key, side="left") - 1; ok = (j >= 0) & (cu[np.maximum(j, 0)] == src["user"])
    g_ = np.full(len(key), np.inf); g_[ok] = (src["ts"][ok] - ct[j[ok]]) / 3600
    return np.searchsorted([0, 6, 12, 24, 48, 96, 192], g_, side="right") - 1
gb_fit, gb_ev = gap_bucket(fit), gap_bucket(test)
ev_ra = rows[ra][::10] // 10
gshare_b = np.bincount(gb_fit[ev_ra], minlength=7) / len(ev_ra)
w_gap = np.repeat((np.bincount(gb_ev, minlength=7) / len(gb_ev) / np.maximum(gshare_b, 1e-12))[gb_fit[ev_ra]], 10)
ones = np.ones(len(y))
base = {"plain": ones, "eval-weighted": w_eval, "gap-weighted": w_gap}
print(f"A = {args.a}\nB = {args.b}\n{n_ev:,} half-B events, {args.n} bootstrap resamples of events")
rng = np.random.default_rng(0)
counts = [np.repeat(rng.multinomial(n_ev, np.full(n_ev, 1 / n_ev)), 10) for _ in range(args.n)]
for name, w in base.items():
    a0, b0 = wauc(pa, y, w), wauc(pb, y, w)
    diffs = np.array([wauc(pb, y, w * c) - wauc(pa, y, w * c) for c in counts])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  {name:14s} A {a0:.5f}  B {b0:.5f}  B-A {b0 - a0:+.5f}  95% interval [{lo:+.5f}, {hi:+.5f}]  "
          f"P(B > A) {np.mean(diffs > 0):.2f}", flush=True)
