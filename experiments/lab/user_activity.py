# Run from the repository root:
#   .venv/bin/python experiments/lab/user_activity.py --seq <cache,...> --extra name=<cache> [...]
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and the caches of the current best
# feature set (same arguments as lab/segment_ctr.py).
#
# User ACTIVITY from the click log's timestamps. The base features see the click history only
# through hist_len (capped at 20), clicks_24h and the time since the last click, and the user's
# event count only through finished fit days. A comparison of another team's evaluation scores
# with ours (they score 0.825 public) showed their per-event level tracks the user's click RATE
# (known clicks per day of history) more strongly than ours, most of all for users with few train
# events. Features, all as of the event (clicks strictly before it, no labels of the event):
# total known clicks, clicks in the last 3 / 7 days, span of the known history, clicks per day
# over the last 20 clicks, mean gap, warm-up (pre-train) clicks, the user's events so far
# (impressions need no labels) and train-period clicks per event so far.
# Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--sets", default="all", help="comma-separated feature subsets to test (see SETS)")
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


def activity(dd, fit):
    """Per event of dd (needs user, ts): activity features from the click log and fit events."""
    H = 3600.0
    u, ts = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64)
    o = np.lexsort((ts, u)); u, ts = u[o], ts[o]
    key = u * 10_000_000_000 + ts
    eu, et = dd["user"].astype(np.int64), dd["ts"].astype(np.int64)
    base = eu * 10_000_000_000
    s = np.searchsorted(key, base, "left"); e = np.searchsorted(key, base + et, "left")
    n = e - s
    def count_since(sec):
        return e - np.maximum(s, np.searchsorted(key, base + np.maximum(et - sec, 0), "left"))
    first = np.where(n > 0, ts[np.minimum(s, len(ts) - 1)], et)
    k20 = np.minimum(n, 20)
    t20 = np.where(n > 0, ts[np.maximum(e - k20, 0)], et)
    span20 = np.maximum((et - t20) / H, 1.0)
    warm = np.searchsorted(key, base + e13.TRAIN_START, "left") - s
    warm = np.maximum(warm, 0)
    # user's fit events strictly before the event (impressions; no labels needed)
    fu, fts = fit["user"].astype(np.int64), fit["ts"].astype(np.int64)
    fo = np.lexsort((fts, fu)); fkey = fu[fo] * 10_000_000_000 + fts[fo]
    n_ev = np.searchsorted(fkey, base + et, "left") - np.searchsorted(fkey, base, "left")
    train_clicks = n - warm
    f = {
        "act_log_known": np.log1p(n),
        "act_log_3d": np.log1p(count_since(3 * 86400)),
        "act_log_7d": np.log1p(count_since(7 * 86400)),
        "act_log_span_h": np.log1p((et - first) / H),
        "act_rate20_per_day": np.log(k20 / span20 * 24 + 1e-3),
        "act_mean_gap_h": np.where(k20 > 1, np.log1p((et - t20) / H / np.maximum(k20 - 1, 1)), np.nan),
        "act_log_warm": np.log1p(warm),
        "act_log_events": np.log1p(n_ev),
        "act_clicks_per_event": np.where(n_ev > 0, (train_clicks + 0.63 * 3) / (n_ev + 3), np.nan),
    }
    return {k: v.astype(np.float32) for k, v in f.items()}

ACT = activity(fit, fit)
SETS = {
    "all": list(ACT),
    "rate": ["act_rate20_per_day", "act_mean_gap_h", "act_log_span_h"],
    "counts": ["act_log_known", "act_log_3d", "act_log_7d", "act_log_warm"],
    "events": ["act_log_events", "act_clicks_per_event"],
}
evrow = rows // 10

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]
for k, v in ACT.items():
    vv = v[evrow]
    print(f"  {k:22s} day-6 mean {np.nanmean(vv[d == 6]):7.3f}  days 2-5 mean {np.nanmean(vv[tr]):7.3f}  "
          f"nan {np.isnan(vv).mean():.3f}", flush=True)

import lightgbm as lgb
def run(label, Xm, nm):
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    print(f"  {label:40s} half-B {auc(pb, y[B]):.5f}  half-A {m.best_score['valid_0']['auc']:.5f}  "
          f"({m.best_iteration} trees)", flush=True)
    return m
print("\nLightGBM (lr 0.02):")
run("current best feature set", X, fn)
for s_ in a.sets.split(","):
    cols = SETS[s_]
    Xs = np.concatenate([X, np.stack([ACT[c][evrow] for c in cols], 1)], 1)
    m = run(f"+ activity [{s_}]", Xs, fn + cols)
    g = sorted(zip(m.feature_importance("gain"), fn + cols), reverse=True)
    print("    by gain: " + ", ".join(f"{n_} #{i + 1}" for i, (_, n_) in enumerate(g) if n_ in cols), flush=True)
