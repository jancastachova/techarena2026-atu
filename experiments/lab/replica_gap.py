# Run from the repository root: .venv/bin/python experiments/lab/replica_gap.py
# REAL VS REPLICA: WHICH SIGNALS ADD MORE ON THE REAL DATA? The calibrated 8-dim replica (scale 1.19, MF 0.81736 vs real
# 0.8175) holds only our mechanisms: fixed taste x 8-dim item vectors + item bias + per-user propensity + softmax over the 10
# candidates. The same features are computed in both worlds from that world's own clicks and its own MF vectors of days 0-5
# (real: 22_vectors / 26_vectors s1, s2; replica: replica8_vectors s0-2), on day-6 events with >= 1 known click. A small
# LightGBM on a base set (projected cosine, item popularity / CTR / impressions, history length, repeat) is trained on half A
# of day 6 and scored on half B; then each feature group is added alone. A group whose lift is clearly larger on the real data
# than on the replica points at a mechanism the model family is missing. (Replica base features cannot be reused: its
# tr_cand columns are copied from the real data.)
import importlib.util
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
auc = e13.roc_auc
C = ROOT / "data/cache"
WORLDS = [("real", C / "13_base_c7_9c47a117c762.npz", [C / "22_vectors_c7.npz", C / "26_vectors_c7_s1.npz", C / "26_vectors_c7_s2.npz"]),
          ("replica", C / "replica8_s1.19.npz", [C / f"replica8_vectors_s{s}.npz" for s in range(3)])]

def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)

def features(base, vecs):
    fit, _, _, _, _ = e13.load_base(str(base))
    H, n = e13.long_history(fit, fit, 100)
    d6 = np.flatnonzero((fit["day"] == 6) & (n > 0))
    h, nn = H[d6].astype(np.int64), n[d6].astype(int)
    m = np.arange(100)[None] < nn[:, None]
    it = fit["item"].reshape(-1, 10)[d6].astype(np.int64)
    y = fit["label"].reshape(-1, 10)[d6].ravel()
    NI = int(max(fit["item"].max(), fit["clk_item"].max())) + 1
    F = {}
    # ---- seed-free statistics of days 0-5
    k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
    clk = np.bincount(fit["clk_item"][k], minlength=NI).astype(np.float64)
    tr = np.repeat(fit["day"] < 6, 10)
    imp = np.bincount(fit["item"][tr], minlength=NI).astype(np.float64)
    pos = np.bincount(fit["item"][tr], fit["label"][tr].astype(np.float64), minlength=NI)
    ctr0 = pos.sum() / imp.sum()
    F["logpop"] = np.log1p(clk)[it]
    F["logimp"] = np.log1p(imp)[it]
    F["ctr"] = ((pos + 20 * ctr0) / (imp + 20))[it]
    F["nhist"] = np.repeat(nn, 10).reshape(-1, 10).astype(np.float64)
    F["rep"] = (it[:, :, None] == np.where(m, h, -1)[:, None, :]).any(2).astype(np.float64)
    # user propensity: share of the user's earlier events (days 0-5 and earlier day 6) with a click
    eu = fit["user"].astype(np.int64)
    o = np.lexsort((fit["ts"], eu)); ev_clk = fit["label"].reshape(-1, 10).max(1)[o].astype(np.float64)
    su = eu[o]; first = np.r_[True, su[1:] != su[:-1]]
    grp = np.cumsum(first) - 1; starts = np.flatnonzero(first)
    cs = np.cumsum(ev_clk); prev_clk = cs - ev_clk - (cs - ev_clk)[starts][grp]
    prev_n = np.arange(len(su)) - starts[grp]
    prop = np.empty(len(su)); prop[o] = (prev_clk + 0.63 * 3) / (prev_n + 3)
    pn = np.empty(len(su)); pn[o] = prev_n
    F["prop"] = np.repeat(prop[d6], 10).reshape(-1, 10); F["prop_n"] = np.repeat(pn[d6], 10).reshape(-1, 10)
    # ---- vector features, averaged over seeds
    acc = {}
    for v in vecs:
        W = np.load(v)["W_6"].astype(np.float64)
        _, U = np.linalg.eigh(W[1:].T @ W[1:]); U = U[:, ::-1]
        Z, Rr = (W @ U[:, :8]).astype(np.float32), (W @ U[:, 8:]).astype(np.float32)
        Zh, Zc = Z[h] * m[:, :, None], Z[it]
        uZh, uZc = unit(Z[h]) * m[:, :, None], unit(Zc)
        sims = np.einsum("ekd,ejd->ekj", uZc, uZh)                     # candidate x history-item cosines
        sims_m = np.where(m[:, None, :], sims, -np.inf)
        g = {}
        g["cos8"] = np.einsum("ed,ekd->ek", unit(Zh.sum(1)), uZc)
        g["max8"] = sims_m.max(2)
        top3 = -np.sort(-sims_m, axis=2)[:, :, :3]
        g["top3"] = np.where(np.isfinite(top3), top3, np.nan).__array__()
        g["top3"] = np.nanmean(g["top3"], axis=2)
        g["meansim"] = np.where(m[:, None, :], sims, 0).sum(2) / nn[:, None]
        g["coh"] = np.repeat(np.linalg.norm(uZh.sum(1), axis=1) / nn, 10).reshape(-1, 10)
        r5 = (np.arange(100)[None] < np.minimum(nn, 5)[:, None])[:, :, None]
        g["rec5"] = np.einsum("ed,ekd->ek", unit((Z[h] * r5).sum(1)), uZc)
        fidx = np.clip(nn[:, None] - 1 - np.arange(5)[None], 0, 99)       # 5 oldest known clicks
        g["old5"] = np.einsum("ed,ekd->ek", unit(Z[np.take_along_axis(h, fidx, 1)].sum(1)), uZc)
        g["resid"] = np.einsum("ed,ekd->ek", unit((Rr[h] * m[:, :, None]).sum(1)), unit(Rr[it]))
        g["resnorm"] = np.linalg.norm(Rr[it], axis=2) / np.maximum(np.linalg.norm(Zc, axis=2), 1e-9)
        g["znorm"] = np.linalg.norm(Zc, axis=2)
        ev = uZc.sum(1, keepdims=True) - uZc
        g["evsim"] = np.einsum("ekd,ekd->ek", uZc, ev) / 9
        g["evrank"] = np.argsort(np.argsort(-g["cos8"], axis=1), axis=1).astype(np.float64)
        for kk, vv in g.items():
            acc[kk] = acc.get(kk, 0) + vv / len(vecs)
    F.update(acc)
    F = {kk: vv.ravel() for kk, vv in F.items()}
    half = np.random.default_rng(0).random(len(d6)) < 0.5
    return F, y, np.repeat(half, 10)

BASE = ["cos8", "logpop", "logimp", "ctr", "nhist", "rep"]
GROUPS = {"propensity": ["prop", "prop_n"], "max/top3 sim": ["max8", "top3"], "meansim+coh": ["meansim", "coh"],
          "recent 5": ["rec5"], "oldest 5": ["old5"], "residual dims": ["resid", "resnorm"], "norm": ["znorm"],
          "event context": ["evsim", "evrank"]}
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1,
         seed=0, num_threads=8)

def fitauc(F, y, half, cols):
    X = np.column_stack([F[c] for c in cols]).astype(np.float32)
    A, B = half, ~half
    rng = np.random.default_rng(1).random(A.sum()) < 0.8
    iA = np.flatnonzero(A)
    dtr = lgb.Dataset(X[iA[rng]], y[iA[rng]]); dva = lgb.Dataset(X[iA[~rng]], y[iA[~rng]])
    b = lgb.train(P, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return auc(b.predict(X[B]), y[B]), b.predict(X[B])

res = {}
for name, base, vecs in WORLDS:
    F, y, half = features(base, vecs)
    yB = y[~half]
    print(f"== {name}: {len(y) // 10} day-6 events, row CTR {y.mean():.4f}", flush=True)
    print("   alone (half B): " + "  ".join(f"{c} {auc(F[c][~half], yB):.4f}" for c in F), flush=True)
    a0, _ = fitauc(F, y, half, BASE)
    allc = BASE + sum(GROUPS.values(), [])
    aall, _ = fitauc(F, y, half, allc)
    print(f"   base {a0:.5f}   all groups {aall:.5f} ({aall - a0:+.5f})", flush=True)
    res[name] = {"base": a0, "all": aall}
    for gname, cols in GROUPS.items():
        a1, _ = fitauc(F, y, half, BASE + cols)
        res[name][gname] = a1
        print(f"   + {gname:16s} {a1:.5f} ({a1 - a0:+.5f})", flush=True)
print("\nlift real minus lift replica (positive = signal the replica does not contain):")
for g in ["all"] + list(GROUPS):
    lr = res["real"][g] - res["real"]["base"]; ls = res["replica"][g] - res["replica"]["base"]
    print(f"   {g:16s} real {lr:+.5f}  replica {ls:+.5f}  gap {lr - ls:+.5f}")
