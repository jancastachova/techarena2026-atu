# Run from the repository root:  .venv/bin/python experiments/lab/stack.py data/cache/lab_members_hops_day6.npz
# Needs a members file written by lab/ensemble_v5.py --save-members (half-A / half-B predictions
# of LightGBM, XGBoost and the MLP on day 6).
#
# The submission combines the three members by an equal-weight rank average. Here the weights are
# fitted on half A (logistic regression on the members' normal-score ranks) and reported on half
# B: global weights, and weights per bucket of the user's known clicks (the members may differ
# most on light users). Also the rank average with the members' event-level part replaced.
import importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from scipy.optimize import minimize


class LogisticRegression:
    """L2-regularised logistic regression (scikit-learn is not installed)."""
    def __init__(self, C=1.0, max_iter=1000):
        self.C, self.max_iter = C, max_iter
    def fit(self, X, y):
        X1 = np.c_[X, np.ones(len(X))]
        def f(w):
            z = X1 @ w
            l = np.logaddexp(0, z) - y * z
            g = X1.T @ (1 / (1 + np.exp(-z)) - y)
            return l.sum() + 0.5 / self.C * (w[:-1] @ w[:-1]), g + np.r_[w[:-1] / self.C, 0]
        w = minimize(f, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B",
                     options={"maxiter": self.max_iter}).x
        self.coef_, self.intercept_ = w[None, :-1], w[-1]
        return self
    def decision_function(self, X):
        return X @ self.coef_[0] + self.intercept_
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
z = np.load(sys.argv[1])
names = ["LGB full (lr .02)", "XGB full", "MLP full"]
row_day = np.repeat(fit["day"], 10)
rows = np.flatnonzero(row_day >= 2)                 # the order of the lab scripts' feature rows
y = fit["label"][rows]
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
hl = hlen[rows // 10].astype(int)
out = {}
for h in "AB":
    idx = z[f"rows_{h}"]
    P = np.stack([z[f"{n}|{h}"] for n in names], 1)
    R = np.stack([ndtri(rankdata(P[:, j]) / (len(P) + 1)) for j in range(3)], 1)
    out[h] = (idx, R, y[idx], hl[idx])
(iA, RA, yA, hA), (iB, RB, yB, hB) = out["A"], out["B"]
# evaluation-weighted AUC (lab/extra_member.py): half-B events reweighted to evaluation.csv's mix
# of known clicks
_t = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))[1]
_, hl_eval = e13.long_history(_t, fit, e13.HIST_LONG)
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def bucket(h):
    b = np.zeros(len(h), int)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(h >= lo) & (h <= hi)] = k
    return b
b6 = bucket(hl[np.repeat(fit["day"], 10)[rows] == 6][::10])
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(b6, minlength=5) / len(b6))
WB = ratio[bucket(hB)]
def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]
    g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
print("eval-weighted half B: " + ", ".join(f"{n} {wauc(RB[:, j], yB, WB):.5f}" for j, n in enumerate(names))
      + f", equal-weight rank average {wauc(RB.sum(1), yB, WB):.5f}")
print(f"members half B: " + ", ".join(f"{n} {auc(RB[:, j], yB):.5f}" for j, n in enumerate(names)))
print(f"equal-weight rank average  half-A {auc(RA.sum(1), yA):.5f}  half-B {auc(RB.sum(1), yB):.5f}")
lr = LogisticRegression(C=1.0, max_iter=1000).fit(RA, yA)
print(f"fitted global weights {np.round(lr.coef_[0], 3)}  half-B {auc(RB @ lr.coef_[0], yB):.5f}")
buckets = [(0, 5), (6, 20), (21, 50), (51, 100)]
def bucket_feats(R, h):
    cols = [R]
    for lo, hi in buckets:
        k = ((h >= lo) & (h <= hi)).astype(float)[:, None]
        cols += [R * k, k]
    return np.concatenate(cols, 1)
lr2 = LogisticRegression(C=1.0, max_iter=2000).fit(bucket_feats(RA, hA), yA)
print(f"weights per known-clicks bucket          half-B {auc(lr2.decision_function(bucket_feats(RB, hB)), yB):.5f}")
for lo, hi in buckets:
    kA, kB = (hA >= lo) & (hA <= hi), (hB >= lo) & (hB <= hi)
    print(f"  known clicks {lo:3d}-{hi:3d}: members half B " +
          ", ".join(f"{auc(RB[kB, j], yB[kB]):.5f}" for j in range(3)) + f", average {auc(RB[kB].sum(1), yB[kB]):.5f}")
