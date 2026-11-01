# Run from the repository root:
#   .venv/bin/python experiments/lab/coimpression.py --seq <cache,...> --extra name=<cache> [...]
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and cached arrays in data/cache.
#
# Rare items are the biggest weak spot by row count: candidates with 1-10 clicks are 66% of all
# rows and score 0.805 vs 0.830 for better-known items. A rare item has few clicks but ~100
# impressions, and one signal in those impressions has never been used: WHICH ITEMS ARE SHOWN
# TOGETHER. If the simulator picks candidates partly by taste, co-shown items share an audience,
# so a rare item's ~900 co-candidates estimate its vector even when it has 3 clicks.
# Item vectors from co-impressions: positive PMI of "shown in the same event" counts, truncated
# SVD (dim 32), as of each day d (events before d only). Features per row: cosine of the
# candidate with the mean of the user's known clicks in that space, the max over the clicks, and
# the candidate's number of co-impressions. Fit days 2-5; day 6 split by event: half A for early
# stopping, AUC on half B.
import argparse, importlib.util, time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--dim", type=int, default=32)
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

cand_all = fit["item"].reshape(-1, 10).astype(np.int64)
iu, ju = np.triu_indices(10, 1)                          # the 45 candidate pairs of an event
CI = np.full((len(rows), 3), np.nan, np.float32)
cinames = ["coimp_cos_mean", "coimp_cos_max", "coimp_log_n"]
for day in range(2, 7):
    t0 = time.time()
    past = np.flatnonzero(fit["day"] < day)
    c = cand_all[past]
    r_ = np.concatenate([c[:, iu].ravel(), c[:, ju].ravel()])
    c_ = np.concatenate([c[:, ju].ravel(), c[:, iu].ravel()])
    C = coo_matrix((np.ones(len(r_), np.float32), (r_, c_)), shape=(n_items, n_items)).tocsr()
    C.sum_duplicates()
    marg = np.asarray(C.sum(1)).ravel()
    total = marg.sum()
    Cc = C.tocoo()
    pmi = np.log(Cc.data * total / (marg[Cc.row] * marg[Cc.col]))
    keep_p = pmi > 0
    P = coo_matrix((pmi[keep_p].astype(np.float32), (Cc.row[keep_p], Cc.col[keep_p])),
                   shape=(n_items, n_items)).tocsr()
    U, S, _ = svds(P, k=a.dim, random_state=0)
    V = (U * np.sqrt(S)[None, :]).astype(np.float32)
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    Vn = np.where(nrm > 0, V / np.maximum(nrm, 1e-12), 0).astype(np.float32)
    now = np.flatnonzero(fit["day"] == day)
    out = np.full((len(now), 10, 3), np.nan, np.float32)
    for s0 in range(0, len(now), 20000):                  # chunked: events x 100 clicks x dim
        e_ = now[s0:s0 + 20000]
        h = hist[e_]; hl = hlen[e_]
        msk = (np.arange(h.shape[1])[None, :] < hl[:, None]).astype(np.float32)
        Hv = Vn[h] * msk[:, :, None]                        # (n, 100, dim)
        mean_v = Hv.sum(1) / np.maximum(hl, 1)[:, None]
        mean_v /= np.maximum(np.linalg.norm(mean_v, axis=1, keepdims=True), 1e-12)
        cv = Vn[cand_all[e_]]                               # (n, 10, dim)
        cos_mean = np.einsum("nkd,nd->nk", cv, mean_v)
        cos_all = np.einsum("nkd,nhd->nkh", cv, Hv)         # (n, 10, 100)
        cos_all = np.where(msk[:, None, :] > 0, cos_all, -2.0)
        has = (hl > 0)[:, None]
        out[s0:s0 + 20000, :, 0] = np.where(has, cos_mean, np.nan)
        out[s0:s0 + 20000, :, 1] = np.where(has, cos_all.max(2), np.nan)
        out[s0:s0 + 20000, :, 2] = np.log1p(marg[cand_all[e_]])
    CI[pos[(now[:, None] * 10 + np.arange(10)).ravel()]] = out.reshape(-1, 3)
    print(f"day {day}: co-impression SVD on {len(past):,} earlier events, {P.nnz:,} positive pairs "
          f"({time.time() - t0:.0f}s)", flush=True)

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]
d6 = d == 6
nclk = np.bincount(fit["clk_item"][fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY], minlength=n_items)
rare = (nclk[fit["item"][rows]] <= 10)[d6]
cm = np.nan_to_num(CI[d6, 0], nan=-1)
print(f"\nday-6 AUC of the co-impression cosine alone: all {auc(cm, y[d6]):.5f}, "
      f"rare candidates (<= 10 clicks) {auc(cm[rare], y[d6][rare]):.5f}")

import lightgbm as lgb
def run(label, Xm, nm):
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    print(f"  {label:40s} half-B {auc(pb, y[B]):.5f}  ({m.best_iteration} trees)", flush=True)
    return m, pb
print("\nLightGBM (lr 0.02):")
_, pb0 = run("current best feature set", X, fn)
m, pb1 = run("+ co-impression similarity", np.concatenate([X, CI], 1), fn + cinames)
rareB = (nclk[fit["item"][rows]] <= 10)[B]
print(f"  rare candidates on half B: {auc(pb0[rareB], y[B][rareB]):.5f} -> {auc(pb1[rareB], y[B][rareB]):.5f}")
g = sorted(zip(m.feature_importance("gain"), fn + cinames), reverse=True)
print("  co-impression features by gain rank: " + ", ".join(f"{n} #{i + 1}" for i, (_, n) in enumerate(g) if n in cinames))
