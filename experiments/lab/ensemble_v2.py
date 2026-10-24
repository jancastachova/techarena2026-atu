# Run from the repository root: .venv/bin/python experiments/lab/ensemble_v2.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and cached arrays in data/cache.
#
# Measures every candidate member against the TRUE current best. The best submission
# (15_avg_13_15.csv) is the mean of two LightGBM pipelines that differ only in their MF seeds
# (13: seeds 0-14, 15: seeds 15-29), so its fit-period analogue is the rank average of LightGBM
# on 13's MF and LightGBM on 15's MF -- not "LightGBM on 13" alone (0.83571).
# Also tests the obvious upgrade: ONE LightGBM on the MF averaged over all 30 seeds.
# Member predictions on half A / half B are cached to data/cache/lab_members_day6.npz so they can
# be recombined without retraining. Fit days 2-5; day 6 split by event: half A = early stopping
# and weights, half B = report.
import importlib.util, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
OUT = CACHE / "lab_members_day6.npz"
NN_OUT = CACHE / "lab_members_nn_day6.npz"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
mf13 = np.load(str(CACHE / "13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
mf15 = np.load(str(CACHE / "15_mf_c7_46f30c940a6a.npz"))["mf_fit"]
mf20 = np.load(str(CACHE / "20_mf_c7_c6b710c438b5.npz"))["mf_fit"]
mf30 = (mf13 + mf15) / 2                       # 30-seed average (both are 15-seed means)
row_day = np.repeat(fit["day"], 10)
keep = np.flatnonzero(row_day >= 2)
rows, X13, names = e13.feature_matrix(fit, mf13, keep)
_, X15, _ = e13.feature_matrix(fit, mf15, keep)
_, X30, _ = e13.feature_matrix(fit, mf30, keep)
_, X20, _ = e13.feature_matrix(fit, mf20, keep)
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

# popularity taste, as of each fit day
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
names_p = names + ["hist_pop", "cand_pop_minus_hist"]
X30p = np.concatenate([X30, POP], 1)


def gauss_rank(Xm, fit_rows):
    out, extra = np.zeros(Xm.shape, np.float32), []
    for j in range(Xm.shape[1]):
        col = Xm[:, j].astype(np.float64)
        nan = np.isnan(col)
        ref = np.sort(col[fit_rows & ~nan])
        q = (np.searchsorted(ref, col, side="left") + np.searchsorted(ref, col, side="right")) / 2
        out[:, j] = np.where(nan, 0.0, ndtri(np.clip((q + 0.5) / (len(ref) + 1), 1e-4, 1 - 1e-4)))
        if nan.any():
            extra.append(nan.astype(np.float32))
    return np.concatenate([out] + [x[:, None] for x in extra], 1) if extra else out


if len(sys.argv) > 1 and sys.argv[1] == "nn":            # torch stage, own process
    import torch, time
    torch.set_num_threads(8)
    Z = gauss_rank(X30p, tr)
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
        print(f"  MLP seed {seed}: half-A {best:.5f}", flush=True)
    np.savez(NN_OUT, A=np.mean(pa_all, 0), B=np.mean(pb_all, 0))
    sys.exit(0)

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

r = lambda p: rankdata(p) / len(p)
members = {}
def add(name, pred):
    members[name] = pred
    print(f"  {name:32s} half-A {auc(pred[0], y[A]):.5f}  half-B {auc(pred[1], y[B]):.5f}", flush=True)

print("members (fit days 2-5, early stopping on half A):", flush=True)
add("LGB on MF13", lgbm(X13, names))
add("LGB on MF15", lgbm(X15, names))
cur = (r(members["LGB on MF13"][0]) + r(members["LGB on MF15"][0]),
       r(members["LGB on MF13"][1]) + r(members["LGB on MF15"][1]))
base_B = auc(cur[1], y[B])
print(f"  {'CURRENT BEST (avg MF13, MF15)':32s} half-A {auc(cur[0], y[A]):.5f}  half-B {base_B:.5f}", flush=True)
add("LGB on MF30", lgbm(X30, names))
add("LGB on MF30 + poptaste lr.02", lgbm(X30p, names_p, 0.02))
add("XGB on MF30 + poptaste", xgbm(X30p))
add("LGB on MF20 (alpha 0.9)", lgbm(X20, names))
subprocess.run([sys.executable, __file__, "nn"], check=True)
z = np.load(NN_OUT)
add("MLP on MF30 + poptaste", (z["A"], z["B"]))
np.savez(OUT, yA=y[A], yB=y[B], **{f"{k}|A": v[0] for k, v in members.items()},
         **{f"{k}|B": v[1] for k, v in members.items()})

print(f"\nagainst the current best (half B {base_B:.5f}):")
def report(label, keys, w=None):
    w = np.ones(len(keys)) if w is None else np.asarray(w, float)
    pa = sum(wi * r(members[k][0]) for wi, k in zip(w, keys))
    pb = sum(wi * r(members[k][1]) for wi, k in zip(w, keys))
    print(f"  {label:44s} half-A {auc(pa, y[A]):.5f}  half-B {auc(pb, y[B]):.5f}  "
          f"vs current {auc(pb, y[B]) - base_B:+.5f}", flush=True)
report("LGB on MF30 alone", ["LGB on MF30"])
report("MF13 + MF15 + MF30", ["LGB on MF13", "LGB on MF15", "LGB on MF30"])
five = ["LGB on MF30", "LGB on MF30 + poptaste lr.02", "XGB on MF30 + poptaste",
        "LGB on MF20 (alpha 0.9)", "MLP on MF30 + poptaste"]
report("5 members on MF30, equal weights", five)
report("5 members + MF13 + MF15, equal weights", five + ["LGB on MF13", "LGB on MF15"])
