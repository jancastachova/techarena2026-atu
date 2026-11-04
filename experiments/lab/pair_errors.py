# Run from the repository root:
#   TEST_DAY=6 .venv/bin/python experiments/lab/pair_errors.py data/cache/box/lab/lab_preds_aud40a_d6.npz
# PAIR-LEVEL ERROR DECOMPOSITION (2026-11-02, external review's "experiment 46"). On the 45 set's half-B predictions:
#   1. how global AUC splits into within-event and cross-event positive/negative pairs (pair counts and AUC of each);
#   2. within-event AUC by finer known-clicks groups;
#   3. every (clicked, unclicked) pair inside an event, grouped by candidate type -- repeat (in the user's known clicks),
#      popularity (item clicks before the day), projected-cosine taste match (40_cosproj8 col 0) -- with each group's share
#      of pairs, misrank rate and share of all misranked pairs (eval-weighted by the event's known-clicks group).
import importlib.util, os, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
TEST_DAY = int(os.environ.get("TEST_DAY", 6))
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
H, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
_, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
z = np.load(sys.argv[1]); key = [k for k in z.files if k.endswith("|B")][0]
r = rows[z["rows_B"]]; p = z[key].astype(np.float64)
o = np.argsort(r, kind="stable"); r, p = r[o], p[o]
y = fit["label"][r]; ev = r[::10] // 10
P = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))).reshape(-1, 10)
Y = y.reshape(-1, 10); I = fit["item"][r].reshape(-1, 10)
COS = np.load(ROOT / "data/cache/40_cosproj8_c7.npz")["mf_fit"][r, 0].reshape(-1, 10)

G5 = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def grp(h, B):
    g = np.full(len(h), -1)
    for k, (lo, hi) in enumerate(B):
        g[(h >= lo) & (h <= hi)] = k
    return g
d_ev = np.flatnonzero(fit["day"] == TEST_DAY)
ratio = (np.bincount(grp(hl_eval.astype(int), G5), minlength=5) / len(hl_eval)) / \
        (np.bincount(grp(hlen[d_ev].astype(int), G5), minlength=5) / len(d_ev))
W = ratio[grp(hlen[ev].astype(int), G5)]                     # event weight (evaluation known-clicks mix)

# 1. global AUC = weighted mix of within-event and cross-event pairs
c = Y.max(1) > 0
pos_w = (W[:, None] * Y).sum(); neg_w = (W[:, None] * (1 - Y)).sum()
within_pairs = (W[c] ** 2 * 9).sum()                          # pair weight = product of the two rows' event weights
tot_pairs = pos_w * neg_w
k_ = Y[c].argmax(1); pp = P[c][np.arange(c.sum()), k_]
win = ((P[c] < pp[:, None]).sum(1) + 0.5 * ((P[c] == pp[:, None]).sum(1) - 1)) / 9
within_auc = float((W[c] ** 2 * win).sum() / (W[c] ** 2).sum())
def wauc(s, y_, w_):
    o_ = np.argsort(s, kind="mergesort"); s, y_, w_ = s[o_], y_[o_], w_[o_]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w_ * y_); wn = np.bincount(g, w_ * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
glob = wauc(P.ravel(), Y.ravel().astype(float), np.repeat(W, 10))
f_in = within_pairs / tot_pairs
cross_auc = (glob - f_in * within_auc) / (1 - f_in)
print(f"day {TEST_DAY}: eval-weighted global AUC {glob:.5f}; within-event pairs are {100 * f_in:.4f}% of all pos/neg pairs "
      f"(within AUC {within_auc:.4f}); cross-event pairs {100 * (1 - f_in):.4f}% (AUC {cross_auc:.5f})")
# cross-event pairs split: positive vs a negative from a CLICKED event, or from a NO-CLICK event
neg_clk = (W[c, None] * (1 - Y[c])).sum(); neg_noclk = (W[~c, None] * np.ones((1, 10))).sum()
s_all = P.ravel(); yy = Y.ravel().astype(bool); ww = np.repeat(W, 10); cc = np.repeat(c, 10)
def auc_vs(negmask):
    m = yy | negmask
    return wauc(s_all[m], yy[m].astype(float), ww[m])
print(f"  positives vs negatives of clicked events: AUC {auc_vs(~yy & cc):.5f} ({100 * neg_clk / neg_w:.1f}% of negatives); "
      f"vs no-click events: {auc_vs(~yy & ~cc):.5f} ({100 * neg_noclk / neg_w:.1f}%)")
for name, keep in (("event level fixed, slate kept", None),):
    pass
# what each negative type costs: global AUC if the negatives of NO-CLICK events were all scored at the bottom
L = P.copy(); L[~c] = P.min() - 1
print(f"  if every no-click event sat below all others: global {wauc(L.ravel(), Y.ravel().astype(float), np.repeat(W, 10)):.5f}")

# 2. within-event AUC by finer known-clicks groups
G = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 35), (36, 50), (51, 100)]
g = grp(hlen[ev].astype(int), G)
share = np.bincount(grp(hl_eval.astype(int), G), minlength=len(G)) / len(hl_eval)
print("\nwithin-event AUC by known clicks (eval share; clicked events in half B; within AUC; mean |margin| of misranks)")
for k, (lo, hi) in enumerate(G):
    m = (g == k)[c]
    if m.sum() < 30: continue
    mis = (P[c][m] < pp[m][:, None]) == False
    print(f"  {lo:>3}-{hi:<3} eval {share[k]:.3f}  n {m.sum():6,}  within {win[m].mean():.4f}")

# 3. pairs inside clicked events, by candidate type
clk_before = np.bincount(fit["item"][np.repeat(fit["day"], 10) < TEST_DAY],
                         weights=fit["label"][np.repeat(fit["day"], 10) < TEST_DAY], minlength=n_items + 1)
Hc = H[ev][c]; Ic = I[c]; Pc = P[c]; Cc = COS[c]; Wc = W[c]
rep = np.array([[it in set(h[h > 0].tolist()) for it in row] for h, row in zip(Hc, Ic)])
pop = clk_before[Ic]
def pbin(x):
    return np.digitize(x, [3.5, 10.5, 30.5, 100.5])               # 0-3 / 4-10 / 11-30 / 31-100 / 101+
PB = ["0-3", "4-10", "11-30", "31-100", "101+"]
n = len(Pc); kk = k_
pos_s = Pc[np.arange(n), kk][:, None]
negm = np.ones_like(Pc, bool); negm[np.arange(n), kk] = False
mis = ((Pc > pos_s) + 0.5 * (Pc == pos_s)) * negm                 # misrank mass of each (pos, neg) pair
pw = np.broadcast_to(Wc[:, None], Pc.shape) * negm
tot_mis = (mis * Wc[:, None]).sum(); tot_pw = pw.sum()
def report(title, keys, labels):
    print(f"\n{title}: share of pairs / misrank rate / share of all misranks")
    for v, lab in zip(np.unique(keys[negm]), labels):
        m = (keys == v) & negm
        sp = pw[m].sum() / tot_pw; rate = (mis * Wc[:, None])[m].sum() / pw[m].sum(); sm = (mis * Wc[:, None])[m].sum() / tot_mis
        print(f"  {lab:34s} {100 * sp:5.1f}%  {rate:.4f}  {100 * sm:5.1f}%")
pos_rep = np.broadcast_to(rep[np.arange(n), kk][:, None], Pc.shape)
report("repeat status (clicked / unclicked candidate in the user's known clicks)",
       pos_rep * 2 + rep, ["neither", "only the unclicked is a repeat", "only the clicked is a repeat", "both repeats"])
pp_bin = np.broadcast_to(pbin(pop[np.arange(n), kk])[:, None], Pc.shape)
report("popularity of the CLICKED candidate (clicks before the day)", pp_bin, PB)
report("popularity of the UNCLICKED candidate", pbin(pop), PB)
more = np.sign(pop - pop[np.arange(n), kk][:, None]) + 1
report("unclicked vs clicked popularity", more, ["unclicked less popular", "equal", "unclicked MORE popular"])
cpos = Cc[np.arange(n), kk][:, None]
dc = Cc - cpos
report("taste match: unclicked cos minus clicked cos (projected cosine)", np.digitize(dc, [-1.0, -0.3, 0.0, 0.3, 1.0]),
       ["< -1.0", "-1.0..-0.3", "-0.3..0", "0..0.3", "0.3..1.0", "> 1.0"])
# misranks the taste match already got right: clicked has the higher cosine but the model still ranks it lower
m_rt = negm & (dc < 0)
print(f"\nmisranks where the clicked candidate HAS the higher taste cosine: {100 * (mis * Wc[:, None])[m_rt].sum() / tot_mis:.1f}% "
      f"of all misranks (rate {(mis * Wc[:, None])[m_rt].sum() / pw[m_rt].sum():.4f})")
# severity: margin of misranked pairs
mg = (Pc - pos_s)[negm & (Pc > pos_s)]
print(f"misranked pairs' logit margin: median {np.median(mg):.3f}, 25% {np.quantile(mg, .25):.3f}, 75% {np.quantile(mg, .75):.3f}; "
      f"near misses (< 0.1) {100 * (mg < 0.1).mean():.1f}%")
# event level, several summaries of the slate (does the event get a click?)
from scipy.special import logsumexp
for name, s in (("mean logit", P.mean(1)), ("sum of probabilities", (1 / (1 + np.exp(-P))).sum(1)),
                ("max logit", P.max(1)), ("logsumexp", logsumexp(P, 1))):
    print(f"event-level AUC, {name:22s} {e13.roc_auc(s, c.astype(float)):.4f}")
