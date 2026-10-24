# Run from the repository root:
#   .venv/bin/python experiments/lab/cluster_affinity.py --seq <cache,...> --extra name=<cache> [...]
# Needs data/cache/22_vectors_c7.npz (22_product_dims --save-vectors: item vectors W_<d> of the MF
# trained on days < d) and the caches of the current best feature set.
#
# lab/simulator_categories.py found no hidden genres (users' clicks concentrate in their top
# k-means cluster only ~1.6x more than chance), but candidates from a user's favourite cluster are
# clicked 28% of the time vs 5% otherwise. A COARSE "how much of this user's history is in the
# candidate's cluster" pools evidence over every item of a cluster -- most useful where item-level
# similarity is noisy (rare candidates, light users). Clusters are fitted per day on W_<d> (MF
# trained only on days before d), so no row sees labels from its own day or later.
# Features per row and resolution k: smoothed share of the user's known clicks in the candidate's
# cluster, and its lift over the cluster's overall click share. Fit days 2-5; day 6 split by
# event: half A for early stopping, AUC on half B.
import argparse, importlib.util
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans2
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--ks", default="20,50,200")
a = ap.parse_args()
KS = [int(x) for x in a.ks.split(",")]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
mf30 = (np.load(str(CACHE / "13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
        + np.load(str(CACHE / "15_mf_c7_46f30c940a6a.npz"))["mf_fit"]) / 2
vec = np.load(str(CACHE / "22_vectors_c7.npz"))
row_day = np.repeat(fit["day"], 10)
keep = np.flatnonzero(row_day >= 2)
rows, X30, names = e13.feature_matrix(fit, mf30, keep)
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))

def group(caches, prefix):
    S = np.mean([np.load(str(c))["mf_fit"] for c in caches.split(",")], 0)[rows]
    nm = [f"{prefix}_{k}" for k in ("logit", "item_bias", "user_bias", "dot",
                                    "minus_max", "rank", "share", "lse")]
    return np.concatenate([S, e13.relative_mf(S)], 1).astype(np.float32), nm

hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
POP = np.zeros((len(rows), 2), np.float32)
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
for spec_ in a.extra:
    name, caches = spec_.split("=", 1)
    g, gn = group(caches, name); parts.append(g); fn += gn
X = np.concatenate(parts, 1)

# cluster-affinity features, as of each day
ALPHA = 2.0                                           # pseudo-clicks toward the cluster's global share
CL = np.full((len(rows), 2 * len(KS)), np.nan, np.float32)
cnames = [f"clu{k}_{w}" for k in KS for w in ("share", "lift")]
for day in range(2, 7):
    W = vec[f"W_{day}"].astype(np.float64)
    before = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
    nclk = np.bincount(fit["clk_item"][before], minlength=n_items)
    ok = nclk >= 5; ok[0] = False
    V = W[ok] / np.linalg.norm(W[ok], axis=1, keepdims=True)
    e = np.flatnonzero(fit["day"] == day)
    h = hist[e]; hl = hlen[e].astype(np.int64)
    valid_h = np.arange(h.shape[1])[None, :] < hl[:, None]
    cand = fit["item"].reshape(-1, 10)[e]
    r_ = pos[(e[:, None] * 10 + np.arange(10)).ravel()]
    for j, k in enumerate(KS):
        _, lab = kmeans2(V, k, seed=day, minit="++", iter=30)
        item_cl = np.full(n_items, -1); item_cl[np.flatnonzero(ok)] = lab
        prior = np.bincount(lab, weights=nclk[ok], minlength=k) / nclk[ok].sum()   # cluster click share
        hc = np.where(valid_h, item_cl[h], -1)
        n_known = (hc >= 0).sum(1).astype(np.float64)
        cc = item_cl[cand]                                                          # (n_ev, 10)
        cnt = np.stack([(hc == cc[:, t:t + 1]).sum(1) for t in range(10)], 1).astype(np.float64)
        pr_c = np.where(cc >= 0, prior[np.maximum(cc, 0)], np.nan)
        share = (cnt + ALPHA * pr_c) / (n_known[:, None] + ALPHA)
        share = np.where(cc >= 0, share, np.nan)
        CL[r_, 2 * j] = share.ravel()
        CL[r_, 2 * j + 1] = np.log(share / pr_c).ravel()
    print(f"day {day}: clusters fitted on {ok.sum():,} items", flush=True)

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]
print(f"\nclick rate by cluster lift on day 6 (k={KS[0]}):")
lift = CL[d == 6, 1]; yy = y[d == 6]
for lo, hi in ((-9, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 9)):
    msk = (lift >= lo) & (lift < hi)
    if msk.sum():
        print(f"  lift [{lo:+.1f},{hi:+.1f}): {msk.mean():6.1%} of rows, click rate {yy[msk].mean():.4f}")

import lightgbm as lgb
def run(label, Xm, nm):
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    print(f"  {label:40s} half-B {auc(pb, y[B]):.5f}  ({m.best_iteration} trees)", flush=True)
    return m
print("\nLightGBM (lr 0.02):")
run("current best feature set", X, fn)
m = run("+ cluster affinity", np.concatenate([X, CL], 1), fn + cnames)
g = sorted(zip(m.feature_importance("gain"), fn + cnames), reverse=True)
print("  cluster features by gain rank: " + ", ".join(f"{n} #{i + 1}" for i, (_, n) in enumerate(g) if n in cnames))
