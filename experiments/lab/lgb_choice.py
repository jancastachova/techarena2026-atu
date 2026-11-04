# Run from the repository root:
#   .venv/bin/python experiments/lab/lgb_choice.py data/cache/lab_export_lean_d6.npz --obj choice --out <preds.npz>
# 2026-11-03: a BOOSTED CHOICE MODEL. The simulator shows 10 candidates and the user clicks at most one: a multinomial
# choice with an outside option. Binary LightGBM scores rows independently; lambdarank / xendcg are shift-invariant per
# event and lose the cross-event level (global AUC is 99.997% cross-event pairs). Here LightGBM gets a custom
# conditional-logit objective: p_i = exp(f_i) / (1 + sum_j exp(f_j)) over the event's candidates, the no-click option
# fixed at score 0, loss -log p(chosen) (or -log p(no click)); gradient p_i - y_i, hessian p_i (1 - p_i). The output p_i is
# a calibrated row probability, comparable across events. Same features and split as lab/lgb_export.py; early stopping
# on half-A AUC of p. --obj binary gives the matched baseline (same learning rate).
import argparse, os, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("export")
ap.add_argument("--obj", choices=["binary", "choice"], default="choice")
ap.add_argument("--out", required=True)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--leaves", type=int, default=63)
ap.add_argument("--threads", type=int, default=int(os.environ.get("NTHREADS", 10)))
ap.add_argument("--extra", default="", help="npz with X / names / rows (lean-export row order) to append as columns")
ap.add_argument("--extra-match", default="", help="keep only extra columns whose name contains this")
a = ap.parse_args()
import lightgbm as lgb
from scipy.stats import rankdata
Z = np.load(a.export)
X, y, A, B, ev = Z["X"], Z["y"].astype(np.float64), Z["A"], Z["B"], Z["ev"]
tr = ~(A | B) & (Z["d"] < Z["d"][A].min())
W_EVAL, FIRST, W_FIRST = Z["W_EVAL"], Z["FIRST"], Z["W_FIRST"]
names = [n.replace(" ", "_") for n in Z["names"]]
if a.extra:
    for f_ in a.extra.split(","):
        E = np.load(f_)
        keep = [j for j, n in enumerate(E["names"]) if a.extra_match in n]
        assert len(E["X"]) == len(X)
        X = np.concatenate([X, E["X"][:, keep]], 1); names += [str(E["names"][j]) for j in keep]
        print(f"  + {len(keep)} extra columns from {f_}", flush=True)
t0 = time.time()


def ev_index(m):
    return np.unique(ev[m], return_inverse=True)[1]


def choice_p(f, g):
    """p_i = exp(f_i) / (1 + sum over the event of exp(f_j)), stable."""
    m = np.zeros(g.max() + 1); np.maximum.at(m, g, f)          # max(0, event max): the no-click score is 0
    ex = np.exp(f - m[g])
    den = np.exp(-m) + np.bincount(g, ex, minlength=len(m))
    return ex / den[g]


P = dict(learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True, force_row_wise=True, num_threads=a.threads)
g_tr, g_A, g_B = ev_index(tr), ev_index(A), ev_index(B)
assert np.all(np.bincount(g_tr) == 10) and np.all(np.bincount(g_A) == 10)
y_tr, y_A = y[tr], y[A]
if a.obj == "binary":
    P.update(objective="binary", metric="auc")
    dtr = lgb.Dataset(X[tr], y_tr, feature_name=names, free_raw_data=True)
    dA = lgb.Dataset(X[A], y_A, reference=dtr)
    m = lgb.train(P, dtr, 8000, valid_sets=[dA], callbacks=[lgb.early_stopping(100, verbose=False)])
    pb = m.predict(X[B], num_iteration=m.best_iteration)
else:
    # init: every candidate at the average row click rate (event click share s, 10 candidates: exp(f) = s/10 / (1 - s))
    s = y_tr.sum() / (len(y_tr) / 10)
    f0 = float(np.log(s / 10 / (1 - s)))

    def fobj(preds, data):
        p = choice_p(preds, g_tr)
        return p - y_tr, np.maximum(p * (1 - p), 1e-9)

    def feval(preds, data):
        r = rankdata(choice_p(preds, g_A)); n1 = y_A.sum()
        return "auc_p", float((r[y_A == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y_A) - n1))), True

    P.update(objective=fobj, metric="None")
    dtr = lgb.Dataset(X[tr], y_tr, init_score=np.full(tr.sum(), f0), feature_name=names, free_raw_data=True)
    dA = lgb.Dataset(X[A], y_A, init_score=np.full(A.sum(), f0), reference=dtr)
    m = lgb.train(P, dtr, 8000, valid_sets=[dA], feval=feval, callbacks=[lgb.early_stopping(100, verbose=False)])
    pb = choice_p(m.predict(X[B], num_iteration=m.best_iteration, raw_score=True) + f0, g_B)


def wauc(s_, y_, w):
    o = np.argsort(s_, kind="mergesort"); s_, y_, w = s_[o], y_[o], w[o]
    new = np.r_[True, s_[1:] != s_[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


yb, fb = y[B], FIRST[B]
print(f"  lgb_choice {os.path.basename(a.export)} obj={a.obj} lr={a.lr}: half-B {wauc(pb, yb, np.ones(len(yb))):.5f}  "
      f"eval-weighted {wauc(pb, yb, W_EVAL[B]):.5f}  first {wauc(pb[fb], yb[fb], W_FIRST[B][fb]):.5f}  "
      f"({m.best_iteration} trees, {time.time() - t0:.0f}s)", flush=True)
np.savez(a.out, rows_B=np.flatnonzero(B), **{"base (= the best member)|B": pb})
print("wrote", a.out)
