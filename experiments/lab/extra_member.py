# Run from the repository root:
#   .venv/bin/python experiments/lab/extra_member.py --seq <cache,...> --extra <name>=<cache,...>
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and cached arrays in data/cache.
#
# Does a further alternative MF (e.g. experiment 21's deep interaction, which scores about as
# well alone as the standalone sequence model) add information ON TOP of the sequence features
# that gave the current best (24_ensemble_seq1.csv, public 0.81853)? The alternative model's 4
# scores plus their event-relative versions are added as LightGBM features next to MF30 +
# popularity taste + sequence features; its rank correlation with the MF and with the sequence
# model is printed first (diversity is what makes such a member useful).
# Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B.
# TEST_DAY=5 (environment) moves the whole split one day earlier: fit days 2-4, day 5 halves (a fresh holdout).
# Also printed: "first", the evaluation-weighted AUC over each user's FIRST event of the test day only (the shape of
# evaluation.csv: one first event per user), weights recomputed for that subset.
# DAYNORM=z (environment, 2026-10-30): MF-derived scores drift with the number of days their MF was trained on (fit day d
# uses an MF of days < d; e.g. cosresid8 col 0 median 0.31 on day 2 -> 0.78 on day 6, evaluation 0.84; lab/adversarial_views.py),
# so the test rows sit beyond the training range. With DAYNORM=z the raw score columns of every MF group (core MF, sequence,
# views; not act_*) and the event log-sum-exp are z-scored within each day (no labels used); event-relative columns are kept.
import argparse, importlib.util, os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True, help="comma-separated 18 --pure-seq caches (averaged)")
ap.add_argument("--extra", action="append", default=[], help="name=cache[,cache...] (averaged)")
ap.add_argument("--base-extra", action="append", default=[],
                help="name=cache[,...]: groups ALWAYS in the base (e.g. the IDF groups of the current best)")
ap.add_argument("--mf-core", default="13_mf_c7_bb1fcb0a9c89.npz,15_mf_c7_46f30c940a6a.npz",
                help="comma-separated MF caches (data/cache) averaged into the core MF features (default MF30)")
ap.add_argument("--save-preds", help="write the half-B predictions of every run to this .npz (lab/paired_ci.py)")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
mf30 = np.mean([np.load(str(CACHE / f))["mf_fit"] for f in a.mf_core.split(",")], 0)
row_day = np.repeat(fit["day"], 10)
keep = np.flatnonzero(row_day >= 2)
rows, X30, names = e13.feature_matrix(fit, mf30, keep)
y, d, ev = fit["label"][rows], row_day[rows], rows // 10

DAYNORM = os.environ.get("DAYNORM", "")
def daynorm(M):
    M = M.astype(np.float64).copy()
    for dd in np.unique(d):
        k = d == dd
        M[k] = (M[k] - M[k].mean(0)) / np.maximum(M[k].std(0), 1e-9)
    return M.astype(np.float32)

def group(caches, prefix):
    S = np.mean([np.load(str(c))["mf_fit"] for c in caches.split(",")], 0)[rows]
    if np.isnan(S).any():
        raise SystemExit(f"{prefix}: scores missing for some rows")
    if S.shape[1] != 4:                        # 2026-11-03: a raw multi-column view (lab/spectral_view.py), no relative columns
        return S.astype(np.float32), [f"{prefix}_{k}" for k in range(S.shape[1])], S.sum(1)
    nm = [f"{prefix}_{k}" for k in ("logit", "item_bias", "user_bias", "dot",
                                    "minus_max", "rank", "share", "lse")]
    R = e13.relative_mf(S)
    if DAYNORM and not prefix.startswith("act"):
        S = daynorm(S); R[:, 3:4] = daynorm(R[:, 3:4])
    return np.concatenate([S, R], 1).astype(np.float32), nm, S[:, 0]

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

if DAYNORM:
    cols_ = [names.index(n_) for n_ in e13.MF_FEATURES + ["mf_event_lse"]]
    X30[:, cols_] = daynorm(X30[:, cols_])
    print(f"DAYNORM={DAYNORM}: MF score columns z-scored within each day", flush=True)
SEQ, snames, s0 = group(a.seq, "seq")
base_parts, base_names = [X30, POP, SEQ], names + ["hist_pop", "cand_pop_minus_hist"] + snames
for spec_ in a.base_extra:
    name, caches = spec_.split("=", 1)
    g, gn, _ = group(caches, name)
    base_parts.append(g); base_names += gn
base_X = np.concatenate(base_parts, 1)
print(f"base: MF30 + popularity + sequence + {[s_.split('=')[0] for s_ in a.base_extra]}")
d6 = d == 6
mf0 = X30[:, names.index("mf_logit")]
corr = lambda u, v: np.corrcoef(rankdata(u[d6]), rankdata(v[d6]))[0, 1]
print(f"sequence ({len(a.seq.split(','))} cache(s)): day-6 AUC {auc(s0[d6], y[d6]):.5f}, "
      f"rank corr with MF30 {corr(s0, mf0):.4f}")
extras = {}
for spec_ in a.extra:
    name, caches = spec_.split("=", 1)
    Xe, en, e0 = group(caches, name)
    extras[name] = (Xe, en)
    print(f"{name} ({len(caches.split(','))} cache(s)): day-6 AUC {auc(e0[d6], y[d6]):.5f}, "
          f"rank corr with MF30 {corr(e0, mf0):.4f}, with sequence {corr(e0, s0):.4f}", flush=True)

TEST_DAY = int(os.environ.get("TEST_DAY", 6))
rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == TEST_DAY]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= TEST_DAY - 1, (d == TEST_DAY) & a_set[ev], (d == TEST_DAY) & ~a_set[ev]
# MINDAY=k (environment, 2026-10-30): train only on fit days k..TEST_DAY-1 (default 2). Early fit days carry MF scores from
# an MF of only d days, far weaker than the 7-day MF behind the evaluation rows.
if os.environ.get("MINDAY"):
    tr = tr & (d >= int(os.environ["MINDAY"]))
    print(f"MINDAY: training on fit days {os.environ['MINDAY']}-{TEST_DAY - 1}", flush=True)
if TEST_DAY != 6:
    print(f"TEST_DAY {TEST_DAY}: fit days 2-{TEST_DAY - 1}, halves of day {TEST_DAY}", flush=True)

# Evaluation-weighted AUC: evaluation.csv holds each user's FIRST event after the cutoff, so light
# users weigh far more there than among day 6's events (1-5 known clicks: 16.4% vs 4.0% of events;
# 51+: 3.6% vs 17.6%). Each half-B event is weighted by (evaluation share / day-6 share) of its
# known-clicks bucket. Uses no labels of evaluation.csv, only its history lengths.
_t = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))[1]
_, hl_eval = e13.long_history(_t, fit, e13.HIST_LONG)
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def bucket(h):
    b = np.zeros(len(h), int)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(h >= lo) & (h <= hi)] = k
    return b
hl_row = hlen[ev].astype(int)
b6 = bucket(hl_row[d == TEST_DAY][::10])
share6 = np.bincount(b6, minlength=len(BUCKETS)) / len(b6)
share_ev = np.bincount(bucket(hl_eval.astype(int)), minlength=len(BUCKETS)) / len(hl_eval)
W_EVAL = (share_ev / np.maximum(share6, 1e-9))[bucket(hl_row)]
_dte = np.flatnonzero(fit["day"] == TEST_DAY); _, _fi = np.unique(fit["user"][_dte], return_index=True)
_first_ev = np.zeros(len(fit["day"]), bool); _first_ev[_dte[_fi]] = True
FIRST = _first_ev[ev]                                   # rows of users' first test-day event (fit events are time-ordered)
_bf = bucket(hl_row[FIRST & (d == TEST_DAY)][::10])
W_FIRST = (share_ev / np.maximum(np.bincount(_bf, minlength=len(BUCKETS)) / len(_bf), 1e-9))[bucket(hl_row)]

def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]
    g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_))
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
print("evaluation-weighted AUC: bucket weights " + ", ".join(
    f"{lo}-{hi} {w_:.2f}" for (lo, hi), w_ in zip(BUCKETS, share_ev / np.maximum(share6, 1e-9))), flush=True)

# Gap-weighted AUC (2026-10-31, lab/gap_profile.py): evaluation users are rarer visitors -- 37% of evaluation events come 48 h+
# after the user's last click vs 11% of day-6 events, and AUC falls with the gap (0.856 at 0-6 h, 0.756 at 192 h+). Half-B
# events reweighted to evaluation's hours-since-last-click mix track the public score (40: 0.8263 vs public 0.8258).
# GAPW=p (environment): also weight TRAINING events by (evaluation share / training share)^p of their gap bucket.
def _gap_h(src):
    cu, ct = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64); o_ = np.lexsort((ct, cu)); cu, ct = cu[o_], ct[o_]
    key = src["user"].astype(np.int64) * 10**10 + src["ts"].astype(np.int64)
    j = np.searchsorted(cu * 10**10 + ct, key, side="left") - 1; ok = (j >= 0) & (cu[np.maximum(j, 0)] == src["user"])
    g_ = np.full(len(key), np.inf); g_[ok] = (src["ts"][ok] - ct[j[ok]]) / 3600; return g_
GBK = [0, 6, 12, 24, 48, 96, 192]
_gb_fit = np.searchsorted(GBK, _gap_h(fit), side="right") - 1; _gb_ev = np.searchsorted(GBK, _gap_h(_t), side="right") - 1
_gshare_ev = np.bincount(_gb_ev, minlength=7) / len(_gb_ev)
_g6 = _gb_fit[np.flatnonzero(fit["day"] == TEST_DAY)]
W_GAP = (_gshare_ev / np.maximum(np.bincount(_g6, minlength=7) / len(_g6), 1e-9))[_gb_fit[ev]]
W_TRAIN = None
if os.environ.get("GAPW"):
    _gt = _gb_fit[np.unique(ev[tr])]
    W_TRAIN = ((_gshare_ev / np.maximum(np.bincount(_gt, minlength=7) / len(_gt), 1e-9)) ** float(os.environ["GAPW"]))[_gb_fit[ev]]
    print(f"GAPW {os.environ['GAPW']}: training event weights by gap bucket " +
          ", ".join(f"{w_:.2f}" for w_ in (_gshare_ev / np.maximum(np.bincount(_gt, minlength=7) / len(_gt), 1e-9)) ** float(os.environ["GAPW"])), flush=True)

if os.environ.get("EXPORT"):                     # EXPORT=path (environment): save the base matrix and split for a torch
    np.savez(os.environ["EXPORT"], X=base_X, y=y, d=d, ev=ev, A=A, B=B, W_EVAL=W_EVAL, FIRST=FIRST, W_FIRST=W_FIRST,
             hl_row=hl_row, names=np.array(base_names))  # member (lab/choice_model.py; no LightGBM in that process)
    raise SystemExit(f"exported {base_X.shape} to {os.environ['EXPORT']}")
import lightgbm as lgb
SAVED = {"rows_B": np.flatnonzero(B)}
def run(label, Xm, nm):
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    if os.environ.get("NTHREADS"):              # NTHREADS=n (environment, 2026-11-01): cap threads so jobs can run side by side
        p["num_threads"] = int(os.environ["NTHREADS"])
    if os.environ.get("LGBP"):                  # LGBP='{"num_leaves": 255}' (environment): LightGBM setting overrides
        import json; p.update(json.loads(os.environ["LGBP"]))
    if os.environ.get("MODEL") == "xgb":         # MODEL=xgb (environment, 2026-11-01): the build's XGBoost member
        import xgboost as xgb                     # (24_ensemble XGB_PARAMS, eta 0.05); XGB_DEVICE=cuda for the GPU
        xp = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", eta=0.05, max_depth=6,
                  min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, seed=0,
                  nthread=int(os.environ.get("NTHREADS", 0)), device=os.environ.get("XGB_DEVICE", "cpu"))
        dva = xgb.DMatrix(Xm[A], y[A])
        if os.environ.get("XGB_RANK"):          # XGB_RANK=G (2026-11-03): pairwise ranking over random groups of G events,
            G = int(os.environ["XGB_RANK"])        # cross-event pairs as in lab/global_rank2.py (a GPU twin of lgbrank)
            ti = np.flatnonzero(tr); ue = np.unique(ev[ti]); np.random.default_rng(0).shuffle(ue)
            ro = np.empty(ue.max() + 1, np.int64); ro[ue] = np.arange(len(ue))
            key = ro[ev[ti]]; o = np.argsort(key, kind="stable"); ti = ti[o]
            xp.update(objective="rank:pairwise", lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=8)
            dtr = xgb.DMatrix(Xm[ti], y[ti], qid=key[o] // G)
            ea = ev[A]; dva = xgb.DMatrix(Xm[A], y[A], qid=np.unique(ea, return_inverse=True)[1] // G)  # one group overflows on the GPU
        else:
            dtr = xgb.DMatrix(Xm[tr], y[tr])
        if os.environ.get("XGB_RANK"):          # GPU ranking AUC overflows int32 on our sizes: fixed rounds (binary picks ~635)
            m = xgb.train(xp, dtr, 700); m.best_iteration = 699
        else:
            m = xgb.train(xp, dtr, 4000, evals=[(dva, "v")], early_stopping_rounds=150, verbose_eval=False)
        m.feature_importance = lambda *_: np.zeros(len(nm))
        pb = m.predict(xgb.DMatrix(Xm[B]), iteration_range=(0, m.best_iteration + 1))
    elif os.environ.get("MODEL") == "cat":         # MODEL=cat (environment, 2026-10-31): CatBoost (oblivious trees) instead of
        from catboost import CatBoost, Pool       # LightGBM, a different tree family for ensemble diversity; CATP = overrides
        import json
        cp = dict(loss_function="Logloss", eval_metric="AUC", iterations=6000, learning_rate=0.08, depth=8,
                  od_type="Iter", od_wait=150, thread_count=-1, verbose=500, random_seed=0)
        cp.update(json.loads(os.environ.get("CATP", "{}")))
        m = CatBoost(cp); m.fit(Pool(Xm[tr], y[tr]), eval_set=Pool(Xm[A], y[A]), use_best_model=True)
        m.best_iteration = m.get_best_iteration(); m.feature_importance = lambda *_: m.get_feature_importance()
        pb = m.predict(Xm[B], prediction_type="RawFormulaVal")
    else:
        m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm, weight=None if W_TRAIN is None else W_TRAIN[tr]), 8000,
                      valid_sets=[lgb.Dataset(Xm[A], y[A], weight=None if W_TRAIN is None else W_GAP[A])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    SAVED[f"{label.strip()}|B"] = pb
    hb = hl_row[B]
    light = (hb >= 1) & (hb <= 20)
    fb = FIRST[B]
    print(f"  {label:44s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  "
          f"first {wauc(pb[fb], y[B][fb], W_FIRST[B][fb]):.5f}  gap-weighted {wauc(pb, y[B], W_GAP[B]):.5f}  "
          f"1-20 known clicks {auc(pb[light], y[B][light]):.5f}  ({m.best_iteration} trees)", flush=True)
    return m
print("\nLightGBM (lr 0.02) on MF30 + popularity taste + sequence, and with each extra model:")
mb = run("base (= the best member)", base_X, base_names)
# LEAN=5,10,20 (environment, 2026-10-30): prune the view groups. Each group's total split gain in the base model ranks the
# --base-extra groups (core MF, popularity and the sequence group always stay); the model is refit on the top-k groups.
if os.environ.get("LEAN"):
    gain = dict(zip(base_names, mb.feature_importance("gain")))
    SUF = ("logit", "item_bias", "user_bias", "dot", "minus_max", "rank", "share", "lse")
    grp = [s_.split("=")[0] for s_ in a.base_extra]
    of = {f"{g_}_{k}": g_ for g_ in grp for k in SUF}         # exact column names, so "cos" never matches "cos_aud_*"
    tot = {g_: sum(v for n_, v in gain.items() if of.get(n_) == g_) for g_ in grp}
    core = sum(v for n_, v in gain.items() if n_ not in of)
    order = sorted(grp, key=lambda g_: -tot[g_])
    allg = sum(gain.values())
    print("group gain share: core+pop+seq %.3f | " % (core / allg) + ", ".join(f"{g_} {tot[g_] / allg:.3f}" for g_ in order), flush=True)
    for k_ in [int(x) for x in os.environ["LEAN"].split(",")]:
        keep_g = set(order[:k_])
        cols = [i for i, n_ in enumerate(base_names) if n_ not in of or of[n_] in keep_g]
        run(f"top {k_} groups ({len(cols)} cols)", base_X[:, cols], [base_names[i] for i in cols])
        print(f"    kept: {order[:k_]}", flush=True)
for name, (Xe, en) in extras.items():
    m = run(f"+ {name}", np.concatenate([base_X, Xe], 1), base_names + en)
    g = sorted(zip(m.feature_importance("gain"), base_names + en), reverse=True)
    top = [f"{n} #{i + 1}" for i, (_, n) in enumerate(g) if n in en][:3]
    print(f"    best-ranked {name} features by gain: " + ", ".join(top))
if a.save_preds:
    np.savez(a.save_preds, **SAVED)
