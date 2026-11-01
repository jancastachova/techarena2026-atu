# Run from the repository root:
#   .venv/bin/python experiments/lab/residuals.py --seq <cache,...> --extra name=<cache> [...]
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and cached arrays in data/cache.
#
# What does the best model still get SYSTEMATICALLY wrong? LightGBM (lr 0.02, fixed 1200 trees,
# no early stopping) is trained on fit days 2-5 with the current best feature set and scores
# all of day 6. Day 6 is split by event into halves A and B; for every user (and every item)
# with rows in both halves, the mean residual (label - prediction) of each half is computed.
# A positive correlation between a user's half-A and half-B residuals means a PERSISTENT user
# effect the features miss -- learnable signal; a correlation near 0 means the leftover error
# is noise. Same for items, and by a few row properties.
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--cross-day", action="store_true",
                help="train on days 2-4 and compare each user's residuals on day 5 with day 6 "
                     "(persistence ACROSS days, which is what the first-event test can use)")
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
for spec_ in a.extra:
    name, caches = spec_.split("=", 1)
    g, gn = group(caches, name); parts.append(g); fn += gn
X = np.concatenate(parts, 1)

import lightgbm as lgb
if a.cross_day:
    trc = d <= 4
    m = lgb.train({**e13.LGB_PARAMS, "learning_rate": 0.02}, lgb.Dataset(X[trc], y[trc], feature_name=fn), 1200)
    users_rows = np.repeat(fit["user"], 10)[rows]
    stats = {}
    for day in (5, 6):
        msk = d == day                                     # whole events, 10 contiguous rows each
        pr_d = m.predict(X[msk])
        print(f"day-{day} AUC (trained on days 2-4): {auc(pr_d, y[msk]):.5f}", flush=True)
        ev_res = (y[msk] - pr_d).reshape(-1, 10).sum(1)   # one residual per event
        ev_user = users_rows[msk].reshape(-1, 10)[:, 0]
        uu, uinv = np.unique(ev_user, return_inverse=True)
        stats[day] = dict(zip(uu.tolist(), np.bincount(uinv, weights=ev_res) / np.bincount(uinv)))
    common = [u for u in stats[5] if u in stats[6]]
    r5 = np.array([stats[5][u] for u in common]); r6 = np.array([stats[6][u] for u in common])
    print(f"\nper-user mean event residual, day 5 vs day 6: {len(common):,} users, "
          f"corr {np.corrcoef(r5, r6)[0, 1]:+.4f} (+-{2 / np.sqrt(len(common)):.4f})")
    raise SystemExit(0)
tr, d6 = d <= 5, d == 6
p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
m = lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=fn), 1200)
pr = m.predict(X[d6])
y6 = y[d6].astype(np.float64)
print(f"day-6 AUC of the model: {auc(pr, y6):.5f}  (fixed 1200 trees, lr 0.02)\n", flush=True)
res = y6 - pr

rng = np.random.default_rng(0)
ev6 = ev[d6]
uev = np.unique(ev6); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(uev, len(uev) // 2, replace=False)] = True
inA = a_set[ev6]

def persistence(keys, label, min_rows=5):
    """Correlation of per-key mean residual between halves A and B (keys present in both)."""
    out = {}
    for half, msk in (("A", inA), ("B", ~inA)):
        k = keys[msk]; r = res[msk]; v = (pr * (1 - pr))[msk]
        uk, inv = np.unique(k, return_inverse=True)
        n = np.bincount(inv); s = np.bincount(inv, weights=r); var = np.bincount(inv, weights=v)
        out[half] = dict(zip(uk.tolist(), zip(n, s / n, var / n ** 2)))
    common = [kk for kk in out["A"] if kk in out["B"] and out["A"][kk][0] >= min_rows
              and out["B"][kk][0] >= min_rows]
    if len(common) < 50:
        print(f"{label}: too few keys in both halves ({len(common)})"); return
    ra = np.array([out["A"][kk][1] for kk in common]); rb = np.array([out["B"][kk][1] for kk in common])
    c = np.corrcoef(ra, rb)[0, 1]
    se = 1 / np.sqrt(len(common))
    # observed variance of per-key mean residuals vs what pure binomial noise would give
    obs = np.var(np.r_[ra, rb])
    exp = np.mean([out["A"][kk][2] for kk in common] + [out["B"][kk][2] for kk in common])
    print(f"{label:34s} keys {len(common):>6,d}  half-to-half corr {c:+.4f} (+-{2*se:.4f})  "
          f"residual variance / binomial noise {obs / exp:.3f}", flush=True)

users = np.repeat(fit["user"], 10)[rows][d6]
items = fit["item"][rows][d6]
print("Persistence of the residuals across the two halves of day 6 (>0 = learnable signal left):")
persistence(users, "per user (rows >= 5 in each half)", 5)
persistence(users, "per user (rows >= 10 in each half)", 10)
persistence(items, "per item (rows >= 5 in each half)", 5)
persistence(items, "per item (rows >= 15 in each half)", 15)
ui = users.astype(np.int64) * 100_000 + items
persistence(ui, "per user x item (rows >= 1)", 1)

# where is the error concentrated? mean |residual| and calibration by a few row properties
print("\nCalibration (mean label / mean prediction) by row property on day 6:")
hl6 = np.repeat(hlen, 10)[rows][d6]
tb6 = ((np.repeat(fit["ts"], 10)[rows][d6] // 3600) % 24) // 6
rank6 = X[d6][:, fn.index("mf_logit_rank")]
for label, keyv, bins in (("known clicks", hl6, [0, 1, 6, 21, 101]),
                          ("time bucket", tb6, [0, 1, 2, 3, 4]),
                          ("MF rank in event", rank6, [1, 2, 3, 6, 11])):
    cells = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        msk = (keyv >= lo) & (keyv < hi)
        if msk.sum():
            cells.append(f"[{lo},{hi}) {y6[msk].mean() / pr[msk].mean():.3f}")
    print(f"  {label:18s} " + "  ".join(cells))
