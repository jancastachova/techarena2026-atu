# Run from the repository root:
#   .venv/bin/python experiments/lab/output_blend.py [data/cache/lab_members_bundle_day6.npz]
# Needs the members file of lab/ensemble_v5.py --save-members (day-6 half-A / half-B predictions of
# the 26_ensemble_bundle members) and the time-forward score caches in data/cache.
#
# Every model so far was combined INSIDE one feature set (LightGBM / XGBoost / MLP on the same ~200
# features, rank-averaged), so the members agree to the fourth decimal. Here structurally different
# STANDALONE models are combined at the OUTPUT instead:
#   step 1  for each standalone model (its own day-6 score from the time-forward caches, mapped to a
#           logit by a 1-variable logistic fit on half A), blend (1 - w) * base logit + w * its logit;
#           w chosen on half A, reported on half B
#   step 2  a stacker: small LightGBM over all member and standalone logits, their within-event rank
#           and gap to the event's best, plus known clicks, user click rate and events; cross-fitted
#           (train on A -> score B, train on B -> score A)
# Base = the members' mean logit (and their rank average, the submitted combination, for reference).
# AUC on half B, plain and evaluation-weighted (lab/extra_member.py), paired bootstrap intervals.
import importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
z = np.load(sys.argv[1] if len(sys.argv) > 1 else str(CACHE / "lab_members_bundle_day6.npz"))
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
iA, iB = z["rows_A"], z["rows_B"]
rA, rB = rows[iA], rows[iB]                                   # absolute row indices
yA, yB = fit["label"][rA].astype(float), fit["label"][rB].astype(float)
assert len(rA) % 10 == 0 and np.all(rA.reshape(-1, 10) // 10 == (rA.reshape(-1, 10) // 10)[:, :1])

# evaluation weights (as lab/extra_member.py)
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
_, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def bucket(h):
    b = np.zeros(len(h), int)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(h >= lo) & (h <= hi)] = k
    return b
ev6 = np.flatnonzero(fit["day"] == 6)
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / \
        (np.bincount(bucket(hlen[ev6].astype(int)), minlength=5) / len(ev6))
wB = ratio[bucket(hlen[rB // 10].astype(int))]
wA = ratio[bucket(hlen[rA // 10].astype(int))]
def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]
    g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
nB = len(yB) // 10
rng = np.random.default_rng(0)
COUNTS = [np.repeat(rng.multinomial(nB, np.full(nB, 1 / nB)), 10) for _ in range(150)]
def ci(base, new, w):
    d = np.array([wauc(new, yB, w * c) - wauc(base, yB, w * c) for c in COUNTS])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"{wauc(new, yB, w) - wauc(base, yB, w):+.5f} [{lo:+.5f}, {hi:+.5f}]"

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
names = ["LGB full (lr .02)", "XGB full", "MLP full"]
MA = np.stack([lg(z[f"{n}|A"]) for n in names], 1); MB = np.stack([lg(z[f"{n}|B"]) for n in names], 1)
baseA, baseB = MA.mean(1), MB.mean(1)
rankB = sum(rankdata(z[f"{n}|B"]) for n in names)
ones = np.ones(len(yB))
print(f"half B: rank average (submitted) {auc(rankB, yB):.5f} / eval-weighted {wauc(rankB, yB, wB):.5f}; "
      f"mean logit {auc(baseB, yB):.5f} / {wauc(baseB, yB, wB):.5f}", flush=True)

def calib(sA, sB):
    """1-variable logistic fit a + b s on half A -> logits for A and B."""
    def f(t):
        q = t[0] + t[1] * sA
        return (np.logaddexp(0, q) - yA * q).sum(), np.array([(1 / (1 + np.exp(-q)) - yA).sum(),
                                                             ((1 / (1 + np.exp(-q)) - yA) * sA).sum()])
    t = minimize(f, np.array([-2.7, 1.0]), jac=True, method="L-BFGS-B").x
    return t[0] + t[1] * sA, t[0] + t[1] * sB
C = lambda f: str(CACHE / f)
VIEWS = {
    "MF30": [C("13_mf_c7_bb1fcb0a9c89.npz"), C("15_mf_c7_46f30c940a6a.npz")],
    "sequence": open("/private/tmp/claude-501/-Users-ivanbibat-Documents-GitHub-techarena2026-atu/5ccba85e-64db-4069-907c-fe9cbd709195/scratchpad/seq.txt").read().strip().split(","),
    "choice": [C("25_mf_c7_3b91a877c2b2.npz")],
    "clicks-only": [C("25_mf_c7_ee7001a259a8.npz"), C("25_mf_c7_3e0d10f600d1.npz")],
    "IDF": [C("25_mf_c7_8849dbfd9b9f.npz"), C("25_mf_c7_f7999fa81611.npz")],
    "impressions-only": [C("25_mf_c7_773e620088b3.npz")],
    "audience (audc)": [C("26s5_audc_c7.npz")],
    "hop2c": [C("26v5_hop2c_c7.npz")],
    "user offsets (uoff10)": [C("25_mf_c7_9226150de4c2.npz")],
    "deep interaction": [C("21_mf_c7_c101aad57f12.npz")],
}
SA, SB = {}, {}
for k, files in VIEWS.items():
    s = np.mean([np.load(f)["mf_fit"][:, 0] for f in files], 0)
    SA[k], SB[k] = calib(s[rA], s[rB])

print("\nstep 1: (1 - w) * base logit + w * standalone logit; w chosen on half A")
grid = np.round(np.arange(0, 0.41, 0.02), 2)
best = {}
for k in VIEWS:
    accA = [wauc((1 - w) * baseA + w * SA[k], yA, wA) for w in grid]
    w = grid[int(np.argmax(accA))]
    sB = (1 - w) * baseB + w * SB[k]
    best[k] = w
    print(f"  {k:22s} alone half-B {auc(SB[k], yB):.5f} | w {w:.2f}: half-B {auc(sB, yB):.5f}  "
          f"eval-weighted {wauc(sB, yB, wB):.5f}  vs base {ci(baseB, sB, wB)}", flush=True)

# all standalones together, weights by logistic regression on half A (members' mean logit + views)
XA = np.column_stack([baseA] + [SA[k] for k in VIEWS]); XB = np.column_stack([baseB] + [SB[k] for k in VIEWS])
def lr_fit(X, y_, w_, l2=1.0):
    X1 = np.c_[X, np.ones(len(X))]
    def f(t):
        q = X1 @ t
        l = w_ * (np.logaddexp(0, q) - y_ * q)
        g = X1.T @ (w_ * (1 / (1 + np.exp(-q)) - y_))
        return l.sum() + 0.5 * l2 * (t[:-1] @ t[:-1]), g + np.r_[l2 * t[:-1], 0]
    return minimize(f, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B").x
for wname, ww in [("plain", np.ones(len(yA))), ("eval-weighted", wA)]:
    t = lr_fit(XA, yA, ww)
    sB = XB @ t[:-1]
    print(f"  all views, logistic weights fitted on half A ({wname}): half-B {auc(sB, yB):.5f}  "
          f"eval-weighted {wauc(sB, yB, wB):.5f}  vs base {ci(baseB, sB, wB)}")
    print("    weights: " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(["base"] + list(VIEWS), t[:-1])), flush=True)

# step 2: LightGBM stacker, cross-fitted between the halves
def event_rel(s):
    g = s.reshape(-1, 10)
    return np.column_stack([(g - g.max(1, keepdims=True)).ravel(),
                            rankdata(-g, axis=1).ravel()])
def stack_X(M, S, r):
    cols = [M, M.mean(1, keepdims=True)] + [S[k][:, None] for k in VIEWS]
    cols += [event_rel(M.mean(1))] + [event_rel(S[k]) for k in VIEWS]
    ev = fit["ev"][r // 10]
    cols += [hlen[r // 10][:, None].astype(float),
             ev[:, e13.EVENT_FEATURES.index("user_rate_logit")][:, None],
             ev[:, e13.EVENT_FEATURES.index("user_log_events")][:, None]]
    return np.column_stack(cols).astype(np.float32)
import lightgbm as lgb
XsA, XsB = stack_X(MA, SA, rA), stack_X(MB, SB, rB)
P = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)
def crossfit(Xa, ya, wa, Xb, rounds=None):
    # inner split of the training half by event for early stopping
    n = len(ya) // 10; m = np.repeat(np.random.default_rng(1).random(n) < 0.8, 10)
    mdl = lgb.train(P, lgb.Dataset(Xa[m], ya[m], weight=wa[m]), 3000,
                    valid_sets=[lgb.Dataset(Xa[~m], ya[~m], weight=wa[~m])],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    return mdl.predict(Xb, num_iteration=mdl.best_iteration), mdl.best_iteration
print("\nstep 2: LightGBM stacker (members + standalone logits + event-relative + known clicks, user rate, events)")
for wname, ww in [("plain", np.ones(len(yA))), ("eval-weighted", wA)]:
    pB, nt = crossfit(XsA, yA, ww, XsB)
    print(f"  trained on A ({wname} loss, {nt} trees): half-B {auc(pB, yB):.5f}  eval-weighted "
          f"{wauc(pB, yB, wB):.5f}  vs base {ci(baseB, pB, wB)}  vs rank average {ci(rankB, pB, wB)}", flush=True)
