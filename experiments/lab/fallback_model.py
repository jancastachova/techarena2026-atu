# Run from the repository root: .venv/bin/python experiments/lab/fallback_model.py
# Needs data/cache/22_vectors_c7.npz (MF vectors of the MF trained on days < d), the MF30 caches and
# data/cache/lab_members_bundle_day6.npz (the best model's day-6 half-B predictions).
#
# A dedicated FALLBACK model for light and cold users (0-5 known clicks: 17.7% of evaluation events).
# Real light-user events are scarce (~160k training rows), so the fallback also trains on SYNTHETIC
# light users: every training event of days 2-5 replayed as if the user had only k known clicks
# (k drawn from 0-5, the k most recent kept). Features are recomputed from the (truncated) history:
#   MF taste match hv . v_c and its cosine, two-hop audience match u2 . C_c and u2 . v_c (u2 = mean of
#   audience-minus-exposure over the history, as 26's hop2c / hop2), repeat flag, popularity taste of
#   the history and the candidate's distance to it, history length, and the gap to the event's best
#   taste match; history-free features are kept as they are (item click rate, exposure, popularity,
#   MF item bias, user click rate and event count, time bucket, weekend).
# Validation on REAL day-6 events of users with 0-5 known clicks (half A early stopping, half B): AUC
# inside the segment vs the best model, and the global evaluation-weighted AUC when the fallback is
# blended into the best model for those users only.
import importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
zv = np.load(CACHE / "22_vectors_c7.npz")
mf30 = (np.load(CACHE / "13_mf_c7_bb1fcb0a9c89.npz")["mf_fit"] + np.load(CACHE / "15_mf_c7_46f30c940a6a.npz")["mf_fit"]) / 2
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
src = (ROOT / "experiments/26_audience/audience.py").read_text()
ns_ = {"e13": e13, "np": np}
exec(src[src.index("def tastes"):src.index("def scores")], ns_)
CI = {k: e13.CAND_FEATURES.index(k) for k in ["item_ctr_logit", "item_log_impr", "item_log_pop_decay", "item_log_clicks"]}
EI = {k: e13.EVENT_FEATURES.index(k) for k in ["user_rate_logit", "user_log_events", "tb_1", "tb_2", "tb_3", "day_type"]}
NAMES = ["n_known", "mf_dot", "mf_cos", "hop2c", "hop2", "rep", "hist_pop", "cand_minus_hist_pop", "dot_minus_max",
         "dot_rank"] + list(CI) + list(EI) + ["mf_item_bias"]
day_cache = {}
def day_ops(day):
    if day not in day_cache:
        W = zv[f"W_{day}"].astype(np.float32)
        A, _, B = ns_["audience"](fit, W.astype(np.float64), day)
        k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
        lp = np.log1p(np.bincount(fit["clk_item"][k], minlength=n_items)).astype(np.float32)
        day_cache[day] = (W, (A - B).astype(np.float32), lp)
    return day_cache[day]

def features(evs, kk):
    """evs: event indices (one day each call), kk: known clicks to keep per event (<= real count)."""
    day = int(fit["day"][evs[0]]); W, Cm, lp = day_ops(day)
    H = hist[evs, :5].copy(); m = np.arange(5)[None] < kk[:, None]; H[~m] = 0   # newest first, k <= 5
    n = kk.astype(np.float32)[:, None]
    hv = (W[H] * m[:, :, None]).sum(1) / np.maximum(n, 1)
    u2 = (Cm[H] * m[:, :, None]).sum(1) / np.maximum(n, 1)
    items = fit["item"].reshape(-1, 10)[evs]
    Wc, Cc = W[items], Cm[items]
    dot = np.einsum("ed,ekd->ek", hv, Wc)
    cos = dot / (np.linalg.norm(hv, axis=1)[:, None] * np.linalg.norm(Wc, axis=2) + 1e-9)
    h2c, h2 = np.einsum("ed,ekd->ek", u2, Cc), np.einsum("ed,ekd->ek", u2, Wc)
    rep = (items[:, :, None] == np.where(m, H, -1)[:, None, :]).any(2).astype(np.float32)
    hp = np.where(kk > 0, (lp[H] * m).sum(1) / np.maximum(kk, 1), np.nan)
    cp = lp[items]
    rows = (evs[:, None] * 10 + np.arange(10)).ravel()
    cand = fit["cand"][rows]; evf = fit["ev"][np.repeat(evs, 10)]
    cols = [np.repeat(kk, 10), dot.ravel(), cos.ravel(), h2c.ravel(), h2.ravel(), rep.ravel(), np.repeat(hp, 10),
            (cp - hp[:, None]).ravel(), (dot - dot.max(1, keepdims=True)).ravel(), rankdata(-dot, axis=1).ravel()]
    cols += [cand[:, i] for i in CI.values()] + [evf[:, i] for i in EI.values()] + [mf30[rows, 1]]
    return np.stack(cols, 1).astype(np.float32), fit["label"][rows].astype(np.float32), rows

rng = np.random.default_rng(0)
Xs, ys = [], []
for day in range(2, 6):
    evs = np.flatnonzero(fit["day"] == day)
    kk = np.minimum(rng.integers(0, 6, len(evs)), hlen[evs])          # synthetic light users
    X, yy, _ = features(evs, kk); Xs.append(X); ys.append(yy)
    real = evs[hlen[evs] <= 5]                                       # and the real ones, as they are
    X, yy, _ = features(real, hlen[real].astype(int)); Xs.append(X); ys.append(yy)
    print(f"day {day}: {len(evs):,} synthetic + {len(real):,} real light events", flush=True)
Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)

# day 6: real light users, split into the usual halves
z = np.load(CACHE / "lab_members_bundle_day6.npz")
rows_all = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
rB = rows_all[z["rows_B"]]; rA = rows_all[z["rows_A"]]
best_B = sum(rankdata(z[f"{k}|B"]) for k in ["LGB full (lr .02)", "XGB full", "MLP full"]) / (3 * len(rB))
evA, evB = np.unique(rA // 10), np.unique(rB // 10)
lightA, lightB = evA[hlen[evA] <= 5], evB[hlen[evB] <= 5]
XA, yA, _ = features(lightA, hlen[lightA].astype(int)); XB, yB, rowsB = features(lightB, hlen[lightB].astype(int))
import lightgbm as lgb
P = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
mdl = lgb.train(P, lgb.Dataset(Xtr, ytr, feature_name=NAMES), 5000, valid_sets=[lgb.Dataset(XA, yA)],
                callbacks=[lgb.early_stopping(150, verbose=False)])
pf = mdl.predict(XB, num_iteration=mdl.best_iteration)
pos = np.searchsorted(rB, rowsB)                                        # light rows inside half B
assert np.array_equal(rB[pos], rowsB)
yb = fit["label"][rB].astype(float)
_, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
bucket = lambda h: np.digitize(h, [1, 6, 21, 51]); e6 = np.flatnonzero(fit["day"] == 6)
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(bucket(hlen[e6].astype(int)), minlength=5) / len(e6))
w = ratio[bucket(hlen[rB // 10].astype(int))]
def wauc(s, y_, w_):
    o = np.argsort(s, kind="mergesort"); s, y_, w_ = s[o], y_[o], w_[o]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w_ * y_); wn = np.bincount(g, w_ * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
print(f"\nfallback model: {mdl.best_iteration} trees; top features by gain: " +
      ", ".join(n_ for _, n_ in sorted(zip(mdl.feature_importance("gain"), NAMES), reverse=True)[:6]))
seg = np.zeros(len(rB), bool); seg[pos] = True
for nm, mk in [("0 known clicks", hlen[rB // 10] == 0), ("1-5 known clicks", (hlen[rB // 10] >= 1) & (hlen[rB // 10] <= 5))]:
    mk = mk & seg
    if mk.sum() and 0 < yb[mk].mean() < 1:
        sub = np.zeros(len(rB)); sub[pos] = pf
        print(f"  {nm:16s} ({mk.sum() // 10:,} events): best model AUC {auc(best_B[mk], yb[mk]):.4f}, fallback {auc(sub[mk], yb[mk]):.4f}")
base = wauc(best_B, yb, w)
print(f"global evaluation-weighted AUC, best model: {base:.5f}")
for a_ in [0.25, 0.5, 0.75, 1.0]:
    s = best_B.copy()
    # put the fallback on the best model's scale inside the segment: map its ranks onto the segment's scores
    seg_scores = np.sort(best_B[pos]); fr = rankdata(pf) / len(pf)
    mapped = seg_scores[np.minimum((fr * len(pf)).astype(int), len(pf) - 1)]
    s[pos] = (1 - a_) * best_B[pos] + a_ * mapped
    print(f"  light users: {1 - a_:.2f} best + {a_:.2f} fallback -> evaluation-weighted {wauc(s, yb, w):.5f} ({wauc(s, yb, w) - base:+.5f})")
