# Run from the repository root (same arguments as lab/extra_member.py, --base-extra only):
#   source experiments/lab/bundle_args.sh
#   .venv/bin/python experiments/lab/warmup_feature.py --seq "$SEQ" "${BUNDLE[@]}"
#
# lab/warmup_propensity.py: users' event rates are stable between the warm-up and train periods
# (log corr 0.77), so the warm-up click INTENSITY divided by the train event rate estimates the click
# propensity (corr 0.545 with the train click rate of users with >= 20 events; half of their own
# events: 0.586). lab/user_activity.py gave LightGBM the parts (clicks per day over the last 20,
# warm-up clicks, events so far) but never this ratio or a pooled estimate. Features, per event, as of
# the event (click histories and event counts only, no labels of the scored event):
#   wu_log_int    log warm-up click intensity: (visible warm-up clicks - 1) / their window
#   wu_log_lam    log train event rate so far, smoothed toward the population rate
#   wu_log_ratio  wu_log_int - wu_log_lam (warm-up propensity)
#   wu_pool       pooled propensity: (train clicks + warm-up clicks + prior) / (train events +
#                 lam * warm-up window + prior), warm-up clicks counted as clicks of pseudo-events
#   wu_pool_logit logit of it, and wu_pool_n its effective number of events
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
DAYS = 86400.0

def warmup_features(dd, fit, p0=0.63, m0=3.0, lam0=1.0, lam_k=1.0):
    u, ts = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64)
    o = np.lexsort((ts, u)); u, ts = u[o], ts[o]
    key = u * 10_000_000_000 + ts
    eu, et = dd["user"].astype(np.int64), dd["ts"].astype(np.int64)
    base = eu * 10_000_000_000
    s = np.searchsorted(key, base, "left"); e = np.searchsorted(key, base + et, "left")
    T0 = e13.TRAIN_START
    w_end = np.searchsorted(key, base + min(T0, 1 << 62), "left")
    w_end = np.minimum(np.maximum(w_end, s), e)               # warm-up clicks known as of the event
    kw = w_end - s
    t_old = np.where(kw > 0, ts[np.minimum(s, len(ts) - 1)], et)
    fu, fts = fit["user"].astype(np.int64), fit["ts"].astype(np.int64)
    fo = np.lexsort((fts, fu)); fu, fts = fu[fo], fts[fo]; fkey = fu * 10_000_000_000 + fts
    fs = np.searchsorted(fkey, base, "left"); n_ev = np.searchsorted(fkey, base + et, "left") - fs
    t_first_ev = np.where(n_ev > 0, fts[np.minimum(fs, len(fts) - 1)], et)
    # warm-up window: oldest visible warm-up click -> the user's first train event (or this event)
    win = np.maximum((np.minimum(t_first_ev, et) - t_old) / DAYS, 1e-3)
    intensity = np.where(kw >= 2, (kw - 1) / win, np.nan)
    elapsed = np.maximum((et - T0) / DAYS, 0.0)
    lam = (n_ev + lam_k * lam0) / (elapsed + lam_k)              # events per day, prior lam0 with weight lam_k days
    k_tr = e - w_end                                              # train-period clicks known as of the event
    pseudo = np.where(kw >= 2, lam * win, 0.0)
    pool = (k_tr + np.maximum(kw - 1, 0) + m0 * p0) / (n_ev + pseudo + m0)
    pool = np.clip(pool, 1e-3, 1 - 1e-3)
    f = {
        "wu_log_int": np.log(intensity + 1e-3),
        "wu_log_lam": np.log(lam),
        "wu_log_ratio": np.log(intensity + 1e-3) - np.log(lam),
        "wu_pool": pool,
        "wu_pool_logit": np.log(pool / (1 - pool)),
        "wu_pool_n": np.log1p(n_ev + pseudo),
    }
    return {k: v.astype(np.float32) for k, v in f.items()}

if __name__ == "__main__":
    WU = warmup_features(fit, fit)
    ev_rows = rows // 10
    import lightgbm as lgb
    def run(label, Xm, nm):
        p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
        m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                      valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
        pb = m.predict(Xm[B], num_iteration=m.best_iteration)
        hb = hl_row[B]; light = (hb >= 1) & (hb <= 20)
        print(f"  {label:32s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  "
              f"1-20 known clicks {auc(pb[light], y[B][light]):.5f}  ({m.best_iteration} trees)", flush=True)
        gain = dict(zip(nm, m.feature_importance("gain")))
        top = sorted(gain, key=gain.get, reverse=True)
        print("    ranks by gain: " + ", ".join(f"{c} {top.index(c) + 1}" for c in nm if c.startswith("wu_")), flush=True)
        return pb
    # event-level check: does the pooled estimate predict "event gets a click" better than user_rate?
    ylab_ev = fit["label"].reshape(-1, 10).max(1)
    ur = base_X[:, base_names.index("user_rate_logit")]
    evB = np.unique(ev_rows[B])
    first_row = np.searchsorted(ev_rows, evB)
    print(f"event-level AUC on half B: user_rate {auc(ur[first_row], ylab_ev[evB]):.5f}  "
          f"wu_pool {auc(WU['wu_pool'][evB], ylab_ev[evB]):.5f}", flush=True)
    for lo, hi in [(0, 5), (6, 20), (21, 10**6)]:
        ne = np.exp(base_X[first_row, base_names.index("user_log_events")]) - 1
        k = (ne >= lo) & (ne <= hi)
        print(f"  train events {lo}-{hi}: {k.sum()} events  user_rate {auc(ur[first_row][k], ylab_ev[evB][k]):.5f}  "
              f"wu_pool {auc(WU['wu_pool'][evB][k], ylab_ev[evB][k]):.5f}", flush=True)
    pa = run("current best", base_X, base_names)
    cols = list(WU)
    F = np.stack([WU[c][ev_rows] for c in cols], 1)
    pb_ = run("+ warm-up propensity", np.concatenate([base_X, F], 1), base_names + cols)
    # paired bootstrap over half-B events (as lab/paired_ci.py)
    evb = ev[B]; _, inv = np.unique(evb, return_inverse=True); ne_ = inv.max() + 1
    rs = np.random.default_rng(0); dp, dw = [], []
    for _ in range(200):
        wt = np.bincount(rs.integers(0, ne_, ne_), minlength=ne_)[inv].astype(float)
        dp.append(wauc(pb_, y[B], wt) - wauc(pa, y[B], wt))
        dw.append(wauc(pb_, y[B], wt * W_EVAL[B]) - wauc(pa, y[B], wt * W_EVAL[B]))
    for nm_, dd_ in (("plain", dp), ("eval-weighted", dw)):
        dd_ = np.array(dd_)
        print(f"  {nm_:14s} diff {dd_.mean():+.5f}  95% [{np.percentile(dd_, 2.5):+.5f}, {np.percentile(dd_, 97.5):+.5f}]  "
              f"P(better) {np.mean(dd_ > 0):.2f}", flush=True)
