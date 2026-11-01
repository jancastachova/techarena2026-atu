# Run from the repository root:
#   .venv/bin/python experiments/lab/ensemble_v4.py data/cache/18_mf_c7_<a>.npz,data/cache/18_mf_c7_<b>.npz,...
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and cached arrays in data/cache.
# Needs: data/cache/lab_members_day6.npz (from lab/ensemble_v2.py) and the cutoff-7 caches of
# several `experiments/18_sequence/sequence.py --pure-seq --mf-seeds 1 --seed-offset N` runs,
# whose sequence scores are AVERAGED (seed averaging, as for the MF).
# Version 4 of ensemble_v3: averaged sequence seeds, and an MLP that also gets the sequence
# features (torch in its own process).
#
# The standalone sequence model (raw item ID sequences; rank correlation 0.860 with the MF on
# day 6) added +0.00046 as LightGBM features on top of 13 (lab/seq_feature.py). Here its 8
# features are added to the best single members of ensemble_v2 (MF averaged over 30 seeds,
# popularity taste), and everything is compared with the TRUE current best: the rank average of
# LightGBM on 13's MF and LightGBM on 15's MF (half B 0.83612).
import importlib.util, subprocess, sys
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
SEQ_CACHES = [Path(c) for c in sys.argv[2 if sys.argv[1:2] == ["nn"] else 1].split(",")]
sq = np.mean([np.load(str(c))["mf_fit"] for c in SEQ_CACHES], 0)
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
XPS = np.concatenate([X30, POP, SEQ], 1)
NN_OUT = CACHE / "lab_members_nnseq_day6.npz"

if sys.argv[1:2] == ["nn"]:                                # torch stage, own process
    import torch
    from scipy.special import ndtri
    torch.set_num_threads(8)
    Z = np.zeros(XPS.shape, np.float32); flags = []
    for j in range(XPS.shape[1]):
        col = XPS[:, j].astype(np.float64); nan = np.isnan(col)
        ref = np.sort(col[tr & ~nan])
        q = (np.searchsorted(ref, col, side="left") + np.searchsorted(ref, col, side="right")) / 2
        Z[:, j] = np.where(nan, 0.0, ndtri(np.clip((q + 0.5) / (len(ref) + 1), 1e-4, 1 - 1e-4)))
        if nan.any():
            flags.append(nan.astype(np.float32))
    if flags:
        Z = np.concatenate([Z] + [f[:, None] for f in flags], 1)
    Zt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype(np.float32))
    idx_tr, idx_A, idx_B = np.flatnonzero(tr), np.flatnonzero(A), np.flatnonzero(B)
    def predict(m, idx):
        m.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(m(Zt[idx[s:s + 65536]])).squeeze(1).numpy()
                                   for s in range(0, len(idx), 65536)])
    pa_all, pb_all = [], []
    for seed in range(3):
        torch.manual_seed(seed); r = np.random.default_rng(seed)
        m = torch.nn.Sequential(
            torch.nn.Linear(Z.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-6)
        best, state, bad = -1.0, None, 0
        for epoch in range(20):
            m.train()
            perm = r.permutation(idx_tr)
            for s in range(0, len(perm), 4096):
                b = torch.from_numpy(perm[s:s + 4096])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Zt[b]).squeeze(1), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            a = auc(predict(m, idx_A), y[A])
            if a > best:
                best, state, bad = a, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 2:
                    break
        m.load_state_dict(state)
        pa_all.append(predict(m, idx_A)); pb_all.append(predict(m, idx_B))
        print(f"  MLP+seq seed {seed}: half-A {best:.5f}", flush=True)
    np.savez(NN_OUT, A=np.mean(pa_all, 0), B=np.mean(pb_all, 0))
    sys.exit(0)

cached = np.load(str(CACHE / "lab_members_day6.npz"))
assert np.array_equal(cached["yB"], y[B]), "cached members were made on a different split"
members = {k.split("|")[0]: (cached[k.split("|")[0] + "|A"], cached[k.split("|")[0] + "|B"])
           for k in cached.files if k.endswith("|A")}
r = lambda p: rankdata(p) / len(p)
cur_B = auc(r(members["LGB on MF13"][1]) + r(members["LGB on MF15"][1]), y[B])
print(f"sequence scores averaged over {len(SEQ_CACHES)} cache(s): " + ", ".join(c.name for c in SEQ_CACHES))
print(f"day-6 sequence AUC {auc(S[d == 6, 0], y[d == 6]):.5f}; rank corr with MF30 "
      f"{np.corrcoef(rankdata(S[d == 6, 0]), rankdata(X30[d == 6, names.index('mf_logit')]))[0, 1]:.4f}")
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
subprocess.run([sys.executable, __file__, "nn", sys.argv[1]], check=True)
z = np.load(NN_OUT)
add("MLP MF30 + pop + seq", (z["A"], z["B"]))

def report(label, keys):
    pb = sum(r(members[k][1]) for k in keys)
    pa = sum(r(members[k][0]) for k in keys)
    print(f"  {label:52s} half-A {auc(pa, y[A]):.5f}  half-B {auc(pb, y[B]):.5f}  "
          f"vs current {auc(pb, y[B]) - cur_B:+.5f}", flush=True)
print("\nequal-weight rank averages:")
report("LGB+pop+seq, XGB+pop+seq", ["LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq"])
report("LGB+seq, LGB+pop+seq, XGB+pop+seq",
       ["LGB MF30 + seq", "LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq"])
report("LGB+pop+seq, XGB+pop+seq, MLP+seq",
       ["LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq", "MLP MF30 + pop + seq"])
report("LGB+pop+seq, XGB+pop+seq, MLP+seq, MLP",
       ["LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq", "MLP MF30 + pop + seq",
        "MLP on MF30 + poptaste"])
if "MLP on MF30 + poptaste" in members:
    report("LGB+pop+seq, XGB+pop+seq, MLP",
           ["LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq", "MLP on MF30 + poptaste"])
    report("all seq members + MLP + LGB MF13 + LGB MF15",
           ["LGB MF30 + seq", "LGB MF30 + pop + seq lr.02", "XGB MF30 + pop + seq",
            "MLP on MF30 + poptaste", "LGB on MF13", "LGB on MF15"])
