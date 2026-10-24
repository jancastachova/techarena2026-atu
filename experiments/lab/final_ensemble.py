# Run from the repository root: .venv/bin/python experiments/lab/final_ensemble.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Needs the MLP day-6 predictions: run experiments/lab/mlp_blend.py first (writes
# data/cache/lab_mlp_day6.npz with the SAME half A / half B split as this script).
#
# Why: five changes each gave about +0.0003 on half B -- noise on their own, but they come from
# different model families (LightGBM, XGBoost, MLP), different features (popularity taste) and a
# different MF (experiment 20's alpha 0.9). Their errors should be partly independent, so an
# EQUAL-WEIGHT rank average may add up. Equal weights are fixed in advance, so nothing is tuned
# on half A and half B stays a clean read. Fit days 2-5; baseline (13's row model) 0.83571.
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
MLP = ROOT / "data" / "cache" / "lab_mlp_day6.npz"
MF20 = ROOT / "data" / "cache" / "20_mf_c7_c6b710c438b5.npz"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
auc = e13.roc_auc
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
row_day = np.repeat(fit["day"], 10)
keep = np.flatnonzero(row_day >= 2)
rows, X, names = e13.feature_matrix(fit, mf, keep)
y, d, ev = fit["label"][rows], row_day[rows], rows // 10

# same split as lab/mlp_blend.py, lab/xgb_blend.py, lab/ease_poptaste.py
rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

# popularity taste, as of each fit day
POP = np.zeros((len(rows), 2), np.float32)
pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))
for day in range(2, 7):
    k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
    lp = np.log1p(np.bincount(fit["clk_item"][k], minlength=n_items)).astype(np.float32)
    e = np.flatnonzero(fit["day"] == day)
    hl = hlen[e].astype(np.float32)
    m = np.arange(hist.shape[1])[None, :] < hl[:, None]
    hp = np.where(hl > 0, np.where(m, lp[hist[e]], 0).sum(1) / np.maximum(hl, 1), np.nan)
    cp = lp[fit["item"].reshape(-1, 10)[e]]
    POP[pos[(e[:, None] * 10 + np.arange(10)).ravel()]] = np.stack(
        [np.repeat(hp, 10), cp.ravel() - np.repeat(hp, 10)], 1)
Xp, names_p = np.concatenate([X, POP], 1), names + ["hist_pop", "cand_pop_minus_hist"]
_, X20, _ = e13.feature_matrix(fit, np.load(str(MF20))["mf_fit"], keep)

import lightgbm as lgb, xgboost as xgb
def lgbm(Xm, nm, lr=0.05):
    p = {**e13.LGB_PARAMS, "learning_rate": lr}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    return m.predict(Xm[A], num_iteration=m.best_iteration), m.predict(Xm[B], num_iteration=m.best_iteration)
def xgbm(Xm):
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", eta=0.05, max_depth=6,
             min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, seed=0, nthread=0)
    dA = xgb.DMatrix(Xm[A], y[A])
    m = xgb.train(p, xgb.DMatrix(Xm[tr], y[tr]), 4000, evals=[(dA, "A")],
                  early_stopping_rounds=100, verbose_eval=False)
    it = (0, m.best_iteration + 1)
    return m.predict(dA, iteration_range=it), m.predict(xgb.DMatrix(Xm[B]), iteration_range=it)

members = {}
members["LGB 13"] = lgbm(X, names)
base_B = auc(members["LGB 13"][1], y[B])
print(f"LGB 13 (baseline)            half-B {base_B:.5f}", flush=True)
for name, p in [("LGB + poptaste, lr 0.02", lgbm(Xp, names_p, 0.02)),
                ("LGB on MF alpha 0.9", lgbm(X20, names)),
                ("XGB + poptaste", xgbm(Xp))]:
    members[name] = p
    print(f"{name:28s} half-B {auc(p[1], y[B]):.5f}", flush=True)
if MLP.exists():
    z = np.load(MLP)
    members["MLP (3 seeds)"] = (z["A"], z["B"])
    print(f"{'MLP (3 seeds)':28s} half-B {auc(z['B'], y[B]):.5f}", flush=True)
else:
    print("MLP predictions missing: run experiments/lab/mlp_blend.py first", flush=True)

r = lambda p: rankdata(p) / len(p)
keys = list(members)
print("\nrank correlations with LGB 13 on half B: " +
      ", ".join(f"{k} {np.corrcoef(r(members['LGB 13'][1]), r(members[k][1]))[0,1]:.4f}"
                for k in keys if k != "LGB 13"))
# logistic stacking: weights fitted on half A over the members' scores, read on half B
from scipy.optimize import minimize
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))

def fit_logistic(F, t, l2):
    """L2-penalised logistic regression (L-BFGS); F already standardised, bias appended."""
    def obj(w):
        z = F @ w
        loss = np.logaddexp(0, z).sum() - (t * z).sum() + 0.5 * l2 * (w[:-1] ** 2).sum()
        g = F.T @ (1 / (1 + np.exp(-z)) - t) + l2 * np.r_[w[:-1], 0.0]
        return loss / len(t), g / len(t)
    w0 = np.zeros(F.shape[1])
    return minimize(obj, w0, jac=True, method="L-BFGS-B").x

def stack(name, fA, fB):
    SA = np.stack([fA(k) for k in keys], 1)
    SB = np.stack([fB(k) for k in keys], 1)
    mu, sd = SA.mean(0), SA.std(0) + 1e-9
    FA = np.c_[(SA - mu) / sd, np.ones(len(SA))]
    FB = np.c_[(SB - mu) / sd, np.ones(len(SB))]
    for l2 in (1.0, 100.0):
        w = fit_logistic(FA, y[A].astype(float), l2)
        pb = FB @ w
        print(f"{name} stacking (l2={l2:g}): half-A {auc(FA @ w, y[A]):.5f}  half-B {auc(pb, y[B]):.5f}"
              f"  gain {auc(pb, y[B]) - base_B:+.5f}")
        print("    weights: " + ", ".join(f"{k} {wi:+.3f}" for k, wi in zip(keys, w[:-1])), flush=True)

stack("log-odds", lambda k: lg(members[k][0]), lambda k: lg(members[k][1]))
stack("rank    ", lambda k: r(members[k][0]), lambda k: r(members[k][1]))
print()
for n in range(2, len(keys) + 1):
    sub = keys[:n]
    eq_A = np.mean([r(members[k][0]) for k in sub], 0)
    eq_B = np.mean([r(members[k][1]) for k in sub], 0)
    print(f"equal-weight average of {n} ({', '.join(sub)}):\n"
          f"    half-A {auc(eq_A, y[A]):.5f}  half-B {auc(eq_B, y[B]):.5f}  "
          f"gain {auc(eq_B, y[B]) - base_B:+.5f}", flush=True)
