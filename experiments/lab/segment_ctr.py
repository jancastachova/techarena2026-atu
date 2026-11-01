# Run from the repository root:
#   .venv/bin/python experiments/lab/segment_ctr.py --seq <cache,...> --extra name=<cache> [...]
# Needs data/cache/22_vectors_c7.npz (item vectors W_<d> of an MF trained on days < d) and the
# caches of the current best feature set.
#
# Click rate of the candidate AMONG USERS WITH SIMILAR TASTE -- a classic CTR-competition feature
# (item x user-segment target encoding) that the pipeline does not have. The base features hold
# each item's OVERALL click rate, which mixes the item's fans with users who do not care for it.
# Here users are grouped by taste (k-means on their taste vector = mean of the MF vectors of
# their known clicks), and for each candidate the click rate among impressions shown to users of
# the SAME group is counted -- a direct count of real outcomes, not a learned approximation.
# Everything is as of each day d: item vectors from an MF trained on days < d, group centroids
# fitted on events before d, click counts from impressions before d. Features: the smoothed group
# click rate (shrunk to the item's overall rate), its lift over the item's overall rate, and the
# group impression count.
# Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B.
import argparse, importlib.util
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans2, vq
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--ks", default="10,30")
ap.add_argument("--smooth", type=float, default=20.0, help="pseudo-impressions toward the item's rate")
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

# ---- item x taste-group click rates, as of each day
item_all = fit["item"].reshape(-1, 10)
lab_all = fit["label"].reshape(-1, 10).astype(np.float64)
hv_cache = {}
def taste(day, events, chunk=20000):
    """Each event's user taste vector (mean of W_<day> over the known clicks, unit length; NaN if
    none). Built in chunks: all events x 100 clicks x 32 dims at once would need ~15 GB."""
    W = vec[f"W_{day}"].astype(np.float32)
    out = np.empty((len(events), W.shape[1]), np.float32)
    for s0 in range(0, len(events), chunk):
        e_ = events[s0:s0 + chunk]
        h = hist[e_]; hl = hlen[e_].astype(np.float32)
        msk = np.arange(h.shape[1])[None, :] < hl[:, None]
        out[s0:s0 + chunk] = (W[h] * msk[:, :, None]).sum(1) / np.maximum(hl, 1)[:, None]
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(nrm > 0, out / np.maximum(nrm, 1e-12), np.nan).astype(np.float64)

SEG = np.full((len(rows), 3 * len(KS)), np.nan, np.float32)
snames = [f"seg{k}_{w}" for k in KS for w in ("ctr_logit", "lift", "log_impr")]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
for day in range(2, 7):
    past = np.flatnonzero(fit["day"] < day)            # labelled events before the scored day
    now = np.flatnonzero(fit["day"] == day)
    vp = taste(day, past); vn = taste(day, now)
    okp = ~np.isnan(vp[:, 0]); okn = ~np.isnan(vn[:, 0])
    imp_i = np.bincount(item_all[past].ravel(), minlength=n_items).astype(np.float64)
    clk_i = np.bincount(item_all[past].ravel(), weights=lab_all[past].ravel(), minlength=n_items)
    ctr_i = (clk_i + 50 * e13.CTR_PRIOR) / (imp_i + 50)
    r_now = pos[(now[:, None] * 10 + np.arange(10)).ravel()]
    for j, k in enumerate(KS):
        sample = vp[okp][np.random.default_rng(day).choice(okp.sum(), min(60000, okp.sum()), replace=False)]
        cent, _ = kmeans2(sample, k, seed=day, minit="++", iter=25)
        seg_p = np.full(len(past), -1); seg_p[okp] = vq(vp[okp], cent)[0]
        seg_n = np.full(len(now), -1); seg_n[okn] = vq(vn[okn], cent)[0]
        key_p = (item_all[past].astype(np.int64) * k + np.repeat(seg_p, 10).reshape(-1, 10))
        valid = np.repeat(seg_p >= 0, 10).reshape(-1, 10)
        imp_s = np.bincount(key_p[valid], minlength=n_items * k).astype(np.float64)
        clk_s = np.bincount(key_p[valid], weights=lab_all[past][valid], minlength=n_items * k)
        cand = item_all[now]
        key_n = cand.astype(np.int64) * k + np.repeat(np.maximum(seg_n, 0), 10).reshape(-1, 10)
        prior = ctr_i[cand]
        rate = (clk_s[key_n] + a.smooth * prior) / (imp_s[key_n] + a.smooth)
        has = np.repeat(seg_n >= 0, 10).reshape(-1, 10)
        SEG[r_now, 3 * j] = np.where(has, lg(rate), np.nan).ravel()
        SEG[r_now, 3 * j + 1] = np.where(has, lg(rate) - lg(prior), np.nan).ravel()
        SEG[r_now, 3 * j + 2] = np.where(has, np.log1p(imp_s[key_n]), np.nan).ravel()
    print(f"day {day}: taste groups fitted on {okp.sum():,} earlier events", flush=True)

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]
d6 = d == 6
print(f"\nday-6 AUC of the group click rate alone (k={KS[0]}): {auc(np.nan_to_num(SEG[d6, 0], nan=-9), y[d6]):.5f}; "
      f"lift alone {auc(np.nan_to_num(SEG[d6, 1], nan=0), y[d6]):.5f}")

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
m = run("+ item click rate within taste group", np.concatenate([X, SEG], 1), fn + snames)
g = sorted(zip(m.feature_importance("gain"), fn + snames), reverse=True)
print("  group features by gain rank: " + ", ".join(f"{n} #{i + 1}" for i, (_, n) in enumerate(g) if n in snames))
