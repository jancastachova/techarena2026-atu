# Run from the repository root:
#   .venv/bin/python experiments/lab/ensemble_v5.py --seq <cache,...> --extra idf=<cache,...> [...]
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and cached arrays in data/cache.
#
# The three-member ensemble of the current best (24_ensemble_seq4.csv, public 0.81891: LightGBM
# + XGBoost + MLP on MF30 + popularity + sequence; half B 0.83701) with every --extra group of
# alternative-MF features added to all three members ("full" in 24_ensemble/ensemble.py).
# Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B.
import argparse, importlib.util, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
NN_OUT = CACHE / "lab_members_nnfull_day6.npz"
ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--ref", type=float, default=0.83850,
                help="half-B AUC of the combination behind the current best upload (for the last line)")
ap.add_argument("--mlp-lr", type=float, default=1e-3)
ap.add_argument("--mlp-seeds", type=int, default=3)
ap.add_argument("--save-members", help="write the members' half-A/half-B predictions to this .npz")
ap.add_argument("--lgb-extra", default="", help="LightGBM settings for the full member, e.g. feature_fraction=0.5")
ap.add_argument("--stage", help=argparse.SUPPRESS)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
mf30 = (np.load(str(CACHE / "13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
        + np.load(str(CACHE / "15_mf_c7_46f30c940a6a.npz"))["mf_fit"]) / 2
row_day = np.repeat(fit["day"], 10)
keep = np.flatnonzero(row_day >= 2)
rows, X30, names = e13.feature_matrix(fit, mf30, keep)
y, d, ev = fit["label"][rows], row_day[rows], rows // 10

def group(caches, prefix):
    S = np.mean([np.load(str(c))["mf_fit"] for c in caches.split(",")], 0)[rows]
    assert not np.isnan(S).any(), prefix
    nm = [f"{prefix}_{k}" for k in ("logit", "item_bias", "user_bias", "dot",
                                    "minus_max", "rank", "share", "lse")]
    return np.concatenate([S, e13.relative_mf(S)], 1).astype(np.float32), nm

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
parts, fn = [X30, POP], names + ["hist_pop", "cand_pop_minus_hist"]
g, gn = group(a.seq, "seq"); parts.append(g); fn += gn
SEQ_ONLY = np.concatenate(parts, 1); seq_names = list(fn)
for spec_ in a.extra:
    name, caches = spec_.split("=", 1)
    g, gn = group(caches, name); parts.append(g); fn += gn
XF = np.concatenate(parts, 1)

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

if a.stage == "nn":                                       # torch stage, own process
    import torch
    torch.set_num_threads(8)
    Z = np.zeros(XF.shape, np.float32); flags = []
    for j in range(XF.shape[1]):
        col = XF[:, j].astype(np.float64); nan = np.isnan(col)
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
    for seed in range(a.mlp_seeds):
        torch.manual_seed(seed); r = np.random.default_rng(seed)
        m = torch.nn.Sequential(
            torch.nn.Linear(Z.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(m.parameters(), lr=a.mlp_lr, weight_decay=1e-6)
        best, state, bad = -1.0, None, 0
        for epoch in range(30):
            m.train()
            perm = r.permutation(idx_tr)
            for s in range(0, len(perm), 4096):
                b = torch.from_numpy(perm[s:s + 4096])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Zt[b]).squeeze(1), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            aa = auc(predict(m, idx_A), y[A])
            if aa > best:
                best, state, bad = aa, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 2:
                    break
        m.load_state_dict(state)
        pa_all.append(predict(m, idx_A)); pb_all.append(predict(m, idx_B))
        print(f"  MLP seed {seed}: half-A {best:.5f}", flush=True)
    np.savez(NN_OUT, A=np.mean(pa_all, 0), B=np.mean(pb_all, 0))
    sys.exit(0)

cached = np.load(str(CACHE / "lab_members_day6.npz"))
r = lambda p: rankdata(p) / len(p)
cur_B = auc(r(cached["LGB on MF13|B"]) + r(cached["LGB on MF15|B"]), y[B])
print(f"sequence caches: {len(a.seq.split(','))}; extra groups: {[s_.split('=')[0] for s_ in a.extra]}")
print(f"true baseline (LGB MF13 + LGB MF15): half-B {cur_B:.5f}; reference combination: {a.ref:.5f}",
      flush=True)

import lightgbm as lgb, xgboost as xgb
def lgbm(Xm, nm, lr=0.02, extra=None):
    p = {**e13.LGB_PARAMS, "learning_rate": lr, **(extra or {})}
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
def add(name, pred):
    members[name] = pred
    print(f"  {name:34s} half-B {auc(pred[1], y[B]):.5f}  vs baseline {auc(pred[1], y[B]) - cur_B:+.5f}",
          flush=True)
print("\nmembers:")
add("LGB seq only (lr .02)", lgbm(SEQ_ONLY, seq_names))
lgb_extra = {k: float(v) if "." in v else int(v) for k, v in (kv.split("=") for kv in a.lgb_extra.split(",") if kv)}
add("LGB full (lr .02)", lgbm(XF, fn, extra=lgb_extra))
add("XGB full", xgbm(XF))
subprocess.run([sys.executable, __file__, "--stage", "nn", "--seq", a.seq,
                "--mlp-lr", str(a.mlp_lr), "--mlp-seeds", str(a.mlp_seeds)] +
               sum([["--extra", s_] for s_ in a.extra], []), check=True)
z = np.load(NN_OUT)
add("MLP full", (z["A"], z["B"]))
if a.save_members:
    np.savez(a.save_members, rows_A=np.flatnonzero(A), rows_B=np.flatnonzero(B),
             **{f"{k}|A": v[0] for k, v in members.items()}, **{f"{k}|B": v[1] for k, v in members.items()})
pa = sum(r(members[k][0]) for k in ("LGB full (lr .02)", "XGB full", "MLP full"))
pb = sum(r(members[k][1]) for k in ("LGB full (lr .02)", "XGB full", "MLP full"))
print(f"\nLGB + XGB + MLP, all full, equal weights: half-A {auc(pa, y[A]):.5f}  half-B {auc(pb, y[B]):.5f}  "
      f"vs baseline {auc(pb, y[B]) - cur_B:+.5f}  vs reference {auc(pb, y[B]) - a.ref:+.5f}")
