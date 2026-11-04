# Run from the repository root:
#   TEST_DAY=6 .venv/bin/python experiments/lab/gap_map.py data/cache/box/lab/lab_preds_aud40a_d6.npz
# WHERE THE AUC IS LOST (2026-11-02, asked: "identify any missing gaps to close the lead"). On the 45 set's half-B predictions,
# evaluation-weighted (events reweighted to evaluation.csv's known-clicks mix): the global AUC, then counterfactuals that fix
# ONE part of the problem and keep the rest, so each line is the most that part could ever give:
#   within-event oracle  : inside each event the clicked candidate gets the event's top score (scores otherwise unchanged)
#   event-level oracle   : each event's 10 scores are shifted so clicked events sit above no-click events by their mean
#                          (the slate's internal order kept)
#   per known-clicks group: both fixes for that group only.
import importlib.util, os, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
_, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def bucket(h):
    b = np.zeros(len(h), int)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(h >= lo) & (h <= hi)] = k
    return b
z = np.load(sys.argv[1]); key = [k for k in z.files if k.endswith("|B")][0]
r = rows[z["rows_B"]]; p = z[key].astype(np.float64)
o = np.argsort(r, kind="stable"); r, p = r[o], p[o]
assert len(r) % 10 == 0 and np.all(r[::10] // 10 == r[9::10] // 10)
y = fit["label"][r].astype(np.float64)
ev = r[::10] // 10
b_ev = bucket(hlen[ev].astype(int))
TEST_DAY = int(os.environ.get("TEST_DAY", 6))
d_ev = np.flatnonzero(fit["day"] == TEST_DAY); bd = bucket(hlen[d_ev].astype(int))
share_eval = np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)
ratio = share_eval / (np.bincount(bd, minlength=5) / len(bd))
w = np.repeat(ratio[b_ev], 10)

def wauc(s, y_, w_):
    o_ = np.argsort(s, kind="mergesort"); s, y_, w_ = s[o_], y_[o_], w_[o_]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w_ * y_); wn = np.bincount(g, w_ * (1 - y_))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))

P, Y = p.reshape(-1, 10), y.reshape(-1, 10)
lg = np.log(np.clip(P, 1e-7, 1 - 1e-7) / (1 - np.clip(P, 1e-7, 1 - 1e-7)))     # work on logits (shifts keep slate order)
clicked = Y.max(1) > 0

def within_fix(L, mask):
    L = L.copy()
    for i in np.flatnonzero(mask & clicked):
        k = Y[i].argmax(); top = L[i].max()
        if L[i, k] < top:
            L[i, k] = top + 1e-6
    return L
def event_fix(L, mask, gap=3.0):
    L = L.copy(); m = L.mean(1, keepdims=True)
    # every event of the group: centre it, then put clicked events at +gap and no-click events at -gap around the
    # group's overall mean level -- a perfect "does the event get a click" part for that group
    lvl = m[mask].mean()
    L[mask] = L[mask] - m[mask] + lvl + np.where(clicked[mask], gap, -gap)[:, None]
    return L
def score(L):
    return wauc(L.ravel(), y, w)

base = score(lg)
allm = np.ones(len(P), bool)
print(f"day {TEST_DAY}, {len(P):,} half-B events, eval-weighted AUC {base:.5f} (plain {e13.roc_auc(p, y):.5f})")
print(f"  within-event oracle, all events : {score(within_fix(lg, allm)):.5f}  (+{score(within_fix(lg, allm)) - base:.4f})")
print(f"  event-level oracle, all events  : {score(event_fix(lg, allm)):.5f}  (+{score(event_fix(lg, allm)) - base:.4f})")
# how well each part works now
ev_auc = e13.roc_auc(lg.mean(1), clicked.astype(float))
inside = np.mean([(lg[i] < lg[i, Y[i].argmax()]).sum() / 9 + 0.5 * ((lg[i] == lg[i, Y[i].argmax()]).sum() - 1) / 9
                  for i in np.flatnonzero(clicked)])
print(f"  now: event-level AUC (mean logit vs any click) {ev_auc:.4f}; within-event AUC {inside:.4f}")
print("\nper known-clicks group (eval share; event click rate; within AUC; event AUC; AUC if this group's within part / "
      "event part / both were perfect)")
for k, (lo, hi) in enumerate(BUCKETS):
    m = b_ev == k
    if m.sum() < 50:
        continue
    cr = clicked[m].mean()
    wi = np.mean([(lg[i] < lg[i, Y[i].argmax()]).sum() / 9 for i in np.flatnonzero(m & clicked)])
    ea = e13.roc_auc(lg[m].mean(1), clicked[m].astype(float))
    a_w = score(within_fix(lg, m)); a_e = score(event_fix(lg, m)); a_b = score(event_fix(within_fix(lg, m), m))
    print(f"  {lo:>3}-{hi:<3} share {share_eval[k]:.3f}  click {cr:.3f}  within {wi:.4f}  event {ea:.4f}   "
          f"within +{a_w - base:.4f}  event +{a_e - base:.4f}  both +{a_b - base:.4f}")
# realistic version: what one more "notch" of quality per group buys -- lift that group's within-event ranking by moving
# its clicked candidate's logit up by delta (a uniformly better model for that group)
print("\nsensitivity: clicked candidate's logit +0.1 in one group (a modestly better model there)")
for k, (lo, hi) in enumerate(BUCKETS):
    m = np.repeat(b_ev == k, 10) & (y > 0)
    L = lg.ravel().copy(); L[m] += 0.1
    print(f"  {lo:>3}-{hi:<3} +{score(L.reshape(-1, 10)) - base:.5f}")
L = lg.ravel().copy(); L[y > 0] += 0.1
print(f"  all     +{score(L.reshape(-1, 10)) - base:.5f}")
# what a global gain needs inside the event: within-event AUC after a uniform lift of the clicked candidate's logit
print("\ntarget: uniform lift of the clicked candidate -> within-event AUC / global eval-weighted AUC")
for dl in (0.02, 0.03, 0.05):
    L = lg.ravel().copy(); L[y > 0] += dl; L = L.reshape(-1, 10)
    wi = np.mean([(L[i] < L[i, Y[i].argmax()]).sum() / 9 for i in np.flatnonzero(clicked)])
    print(f"  +{dl:.2f}: within {wi:.4f} (now {inside:.4f})  global +{score(L) - base:.5f}")
m0 = b_ev == 0
ties = np.mean([(lg[i] == lg[i, Y[i].argmax()]).sum() - 1 for i in np.flatnonzero(m0 & clicked)])
print(f"empty-history events: mean ties with the clicked candidate {ties:.2f} of 9")
