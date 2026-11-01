# Run from the repository root: .venv/bin/python experiments/lab/ensemble_v3.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and cached arrays in data/cache.
# Needs: data/cache/lab_members_day6.npz (from lab/ensemble_v2.py) and the pure-sequence scores
# 18_mf_c7_3a4b8237d50b.npz (experiments/18_sequence/sequence.py --cutoff 7 --pure-seq --mf-seeds 1).
#
# The standalone sequence model (raw item ID sequences; rank correlation 0.860 with the MF on
# day 6) added +0.00046 as LightGBM features on top of 13 (lab/seq_feature.py). Here its 8
# features are added to the best single members of ensemble_v2 (MF averaged over 30 seeds,
# popularity taste), and everything is compared with the TRUE current best: the rank average of
# LightGBM on 13's MF and LightGBM on 15's MF (half B 0.83612).
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
mf30 = (np.load(str(CACHE / "13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
        + np.load(str(CACHE / "15_mf_c7_46f30c940a6a.npz"))["mf_fit"]) / 2
sq = np.load(str(CACHE / "18_mf_c7_3a4b8237d50b.npz"))["mf_fit"]
row_day = np.repeat(fit["day"], 10)
keep = np.flatnonzero(row_day >= 2)
rows, X30, names = e13.feature_matrix(fit, mf30, keep)
y, d, ev = fit["label"][rows], row_day[rows], rows // 10

hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
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
S = sq[rows]
assert not np.isnan(S).any()
SEQ = np.concatenate([S, e13.relative_mf(S)], 1)
snames = ["seq_logit", "seq_item_bias", "seq_user_bias", "seq_dot",
          "seq_minus_max", "seq_rank", "seq_share", "seq_lse"]
pnames = ["hist_pop", "cand_pop_minus_hist"]

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

cached = np.load(str(CACHE / "lab_members_day6.npz"))
assert np.array_equal(cached["yB"], y[B]), "cached members were made on a different split"
members = {k.split("|")[0]: (cached[k.split("|")[0] + "|A"], cached[k.split("|")[0] + "|B"])
           for k in cached.files if k.endswith("|A")}
r = lambda p: rankdata(p) / len(p)
cur_B = auc(r(members["LGB on MF13"][1]) + r(members["LGB on MF15"][1]), y[B])
print(f"current best (LGB MF13 + LGB MF15): half-B {cur_B:.5f}", flush=True)

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
def add(name, pred):
    members[name] = pred
    print(f"  {name:36s} half-B {auc(pred[1], y[B]):.5f}  vs current {auc(pred[1], y[B]) - cur_B:+.5f}",
          flush=True)

print("\nnew members with the sequence features:", flush=True)
add("LGB MF30 + seq", lgbm(np.concatenate([X30, SEQ], 1), names + snames))
add("LGB MF30 + pop + seq lr.02", lgbm(np.concatenate([X30, POP, SEQ], 1), names + pnames + snames, 0.02))
add("XGB MF30 + pop + seq", xgbm(np.concatenate([X30, POP, SEQ], 1)))
np.savez(CACHE / "lab_members_seq_day6.npz", **{f"{k}|A": v[0] for k, v in members.items()},
         **{f"{k}|B": v[1] for k, v in members.items()})

def report(label, keys):
    pb = sum(r(members[k][1]) for k in keys)
    pa = sum(r(members[k][0]) for k in keys)
    print(f"  {label:52s} half-A {auc(pa, y[A]):.5f}  half-B {auc(pb, y[B]):.5f}  "
          f"vs current {auc(pb, y[B]) - cur_B:+.5f}", flush=True)
print("\nequal-weight rank averages:")
report("LGB+pop+seq, XGB+pop+seq", ["LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq"])
report("LGB+seq, LGB+pop+seq, XGB+pop+seq",
       ["LGB MF30 + seq", "LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq"])
if "MLP on MF30 + poptaste" in members:
    report("LGB+pop+seq, XGB+pop+seq, MLP",
           ["LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq", "MLP on MF30 + poptaste"])
    report("all seq members + MLP + LGB MF13 + LGB MF15",
           ["LGB MF30 + seq", "LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq",
            "MLP on MF30 + poptaste", "LGB on MF13", "LGB on MF15"])
