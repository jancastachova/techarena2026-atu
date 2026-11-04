# Run from the repository root: TEST_DAY=6 .venv/bin/python experiments/lab/forensic_audit.py
# 2026-11-03 forensic review (reviewer: "audit the simulator's mechanisms before another model"). On half B of day T, with the
# lean export's features and the matched binary LightGBM (lab/lgb_choice.py --obj binary, a calibrated row probability):
#  1. MECHANISM CALIBRATION. Within a user, click share RISES with time since the last click (docs/dataset/numbers.json:
#     -0.008 at < 30 min ... +0.081 at 72-168 h), against a memoryless propensity. Is the model's event-level prediction
#     (sum of p: clicks are exclusive) calibrated by gap x user-rate tercile, and by the number of slate candidates already in the history?
#  2. EXPLICIT EVENT MODEL. P(event has a click) from a separate LightGBM on event aggregates (mean / max / min over the 10
#     candidates of every feature + the model's own event logit), 2-fold cross-fitted inside half B by user. Event-level AUC
#     vs the model's implied one, then the re-levelled row score P_event x p_i / sum p (global, eval-weighted).
#  3. CLUSTERING. Paired bootstrap intervals resampling EVENTS (as lab/paired_ci.py) vs resampling USERS (a user has several
#     day-T events), for gch vs binary (lab/lgb_choice.py) and for the re-levelled score vs base.
import importlib.util, os
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
T = int(os.environ.get("TEST_DAY", 6))
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
Z = np.load(f"data/cache/lab_export_lean_d{T}.npz")
B = Z["B"]; rB = np.flatnonzero(B)
X = Z["X"][B]; names = list(Z["names"]); y = Z["y"][B].astype(float); w = Z["W_EVAL"][B]
pb = np.load(f"data/cache/lab_preds_lgbc_binary_d{T}.npz"); assert np.array_equal(pb["rows_B"], rB)
pg = np.load(f"data/cache/lab_preds_lgbc_choice_d{T}.npz")
p = pb["base (= the best member)|B"]; g = pg["base (= the best member)|B"]
n_ev = len(y) // 10
fev = rows[rB][::10] // 10                                  # fit event index of each half-B event
user = fit["user"][fev]
ye = y.reshape(-1, 10).sum(1)
pe = np.clip(p.reshape(-1, 10).sum(1), 1e-4, 1 - 1e-4)       # model's P(event has a click): clicks are exclusive
we = w[::10]
col = {n: X[:, i].reshape(-1, 10)[:, 0] for i, n in enumerate(names)}   # event-level columns (same on all 10 rows)


def wauc(s, y_, w_):
    o = np.argsort(s, kind="mergesort"); s, y_, w_ = s[o], y_[o], w_[o]
    new = np.r_[True, s[1:] != s[:-1]]; gg = np.cumsum(new) - 1
    wp = np.bincount(gg, w_ * y_); wn = np.bincount(gg, w_ * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


print(f"### day {T} half B: {n_ev:,} events, {len(np.unique(user)):,} users; binary row AUC eval-weighted {wauc(p, y, w):.5f}")
# ---- 1. mechanism calibration (eval-weighted event counts)
hrs = np.expm1(col["log_hours_since_last"]) if col["log_hours_since_last"].max() < 20 else col["log_hours_since_last"]
gap_b = np.searchsorted([0.5, 2, 6, 24, 72, 168], hrs, side="right")
no_h = col["no_hist"] > 0.5
gap_b[no_h] = 7
ur = col["user_rate_logit"]; terc = np.searchsorted(np.quantile(ur, [1 / 3, 2 / 3]), ur, side="right")
print("1a. event click share actual / predicted (ratio), by hours since last click x user-rate tercile:")
lab = ["<.5h", ".5-2h", "2-6h", "6-24h", "1-3d", "3-7d", "7d+", "none"]
for b in range(8):
    cells = []
    for t in range(3):
        m = (gap_b == b) & (terc == t)
        if m.sum() < 200:
            cells.append(f"{'':>22s}"); continue
        a_, p_ = np.average(ye[m] > 0, weights=we[m]), np.average(pe[m], weights=we[m])
        cells.append(f"{a_:.3f}/{p_:.3f} ({a_ / p_:.3f}) n{m.sum():>5d}")
    print(f"  {lab[b]:6s} " + "  ".join(cells))
nrep = X[:, names.index("rep_any")].reshape(-1, 10).sum(1)
print("1b. by slate candidates already in the user's history (event share actual / predicted; row CTR of those candidates):")
rep_row = X[:, names.index("rep_any")] > 0.5
for k in range(4):
    m = (np.minimum(nrep, 3) == k)
    mr = np.repeat(m, 10) & rep_row
    extra = f"  repeat rows CTR {y[mr].mean():.3f} vs pred {p[mr].mean():.3f}" if mr.sum() else ""
    print(f"  {k}{'+' if k == 3 else ' '} history items: events {m.sum():>6d}  {np.average(ye[m] > 0, weights=we[m]):.3f} / "
          f"{np.average(pe[m], weights=we[m]):.3f}{extra}")

# ---- 2. explicit event model, 2-fold by user
import lightgbm as lgb
E = X.reshape(n_ev, 10, -1)
F = np.concatenate([E.mean(1), E.max(1), E.min(1), np.log(pe / (1 - pe))[:, None]], 1)
uu = np.unique(user); rng = np.random.default_rng(0); fold_u = rng.integers(0, 2, len(uu))
fold = fold_u[np.searchsorted(uu, user)]
pe2 = np.zeros(n_ev)
P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0, num_threads=8)
for k in range(2):
    tr_, te_ = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    inner = rng.random(len(tr_)) < 0.2
    m = lgb.train(P, lgb.Dataset(F[tr_[~inner]], ye[tr_[~inner]], init_score=np.log(pe / (1 - pe))[tr_[~inner]]), 3000,
                  valid_sets=[lgb.Dataset(F[tr_[inner]], ye[tr_[inner]], init_score=np.log(pe / (1 - pe))[tr_[inner]])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    pe2[te_] = 1 / (1 + np.exp(-(m.predict(F[te_], raw_score=True) + np.log(pe / (1 - pe))[te_])))
    print(f"2. fold {k}: {m.best_iteration} trees of event-level correction")
print(f"2. event-level AUC (eval-weighted): model implied {wauc(pe, ye, we):.5f}  explicit event model {wauc(pe2, ye, we):.5f}")
share = (p.reshape(-1, 10) / p.reshape(-1, 10).sum(1, keepdims=True)).reshape(-1)
s_new = np.repeat(pe2, 10) * share
s_mid = np.repeat(np.sqrt(pe2 * pe), 10) * share
print(f"   global eval-weighted: base {wauc(p, y, w):.5f}  re-levelled {wauc(s_new, y, w):.5f}  half-way {wauc(s_mid, y, w):.5f}")

# ---- 3. event- vs user-clustered intervals
ev_of_row = np.repeat(np.arange(n_ev), 10)
uid = np.searchsorted(uu, user)
def boot(a, b, by_user, n=150):
    r = np.random.default_rng(1); out = []
    for _ in range(n):
        if by_user:
            c = r.multinomial(len(uu), np.full(len(uu), 1 / len(uu)))[uid]
        else:
            c = r.multinomial(n_ev, np.full(n_ev, 1 / n_ev))
        cw = w * c[ev_of_row]
        out.append(wauc(b, y, cw) - wauc(a, y, cw))
    return np.percentile(out, [2.5, 97.5])
for nm, a_, b_ in (("gch vs binary", p, g), ("re-levelled vs base", p, s_new)):
    d0 = wauc(b_, y, w) - wauc(a_, y, w)
    e_lo, e_hi = boot(a_, b_, False); u_lo, u_hi = boot(a_, b_, True)
    print(f"3. {nm:20s} {d0:+.5f}  events [{e_lo:+.5f}, {e_hi:+.5f}]  users [{u_lo:+.5f}, {u_hi:+.5f}]", flush=True)
