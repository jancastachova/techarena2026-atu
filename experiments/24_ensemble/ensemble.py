#!/usr/bin/env python3
"""
TechArena CTR -- experiment 24: ensemble of second-stage models over the cached MF scores of
experiments 13, 15 and 20.

Why: the current best (15_avg_13_15.csv, public 0.81807) is the mean of two LightGBM pipelines
that differ only in their MF seeds. On the fit-period check (experiments/lab/ensemble_v2.py),
members trained on different model families, features and MFs add up a little because their
errors are partly independent. This script trains the chosen members on the official split and
writes the rank average of their predictions for evaluation.csv.

Members (choose with --members, comma-separated):
    lgb_mf13      LightGBM (13's settings) on 13's MF (seeds 0-14)        = experiment 13
    lgb_mf15      LightGBM on 15's MF (seeds 15-29)                          = experiment 15
    lgb_mf30      LightGBM on the MF averaged over all 30 seeds
    lgb_pop       LightGBM, learning rate 0.02, on MF30 + popularity taste
    xgb_pop       XGBoost (depth 6) on MF30 + popularity taste
    lgb_mf20      LightGBM on experiment 20's MF (user vector sum / n^0.9)
    mlp_pop       3-layer MLP on MF30 + popularity taste (Gaussian-rank inputs, 3 seeds)
    lgb_seq       LightGBM on MF30 + the standalone sequence model's 8 features
    lgb_popseq    LightGBM, learning rate 0.02, on MF30 + popularity taste + sequence features
    xgb_popseq    XGBoost on MF30 + popularity taste + sequence features
    mlp_popseq    MLP on MF30 + popularity taste + sequence features
Sequence features = the scores of experiment 18's standalone sequence model (raw item ID
sequences, --pure-seq), averaged over the caches given with --seq-caches, plus their
event-relative versions (gap to the best, rank, share, log-sum-exp), as for the MF.
Sequence members need --seq-caches (comma-separated cutoff-7 caches of 18's --pure-seq runs).
    lgb_full      LightGBM, learning rate 0.02, on MF30 + popularity + sequence + every --extra group
    xgb_full      XGBoost on the same
    mlp_full      MLP on the same
--extra name=cache[,cache...] (repeatable) adds a further alternative MF as 8 features (its 4
scores, averaged over the given caches, plus event-relative versions), e.g. experiment 25's
IDF-pooled MF: --extra idf=data/cache/25_mf_c7_<...>.npz
Popularity taste = mean log-popularity of the user's known clicks, and candidate minus that,
as of each day (for the evaluation rows: as of the cutoff).

Procedure per tree member (the same as 13's lgbm_procedure): trees chosen by early stopping on
fit day 6 (trained on days 2-5), then refit on days 2-6 with the tree count scaled by 5/4.
The MLP: epochs chosen the same way, then retrained on days 2-6. Only ranks are combined, so
the output is a rank percentile in [0, 1] (the metric only uses the ranking).

Inputs: the base cache of 13 (cutoff 7) and the MF caches of 13, 15 and 20 in data/cache
(produced by those experiments' --cutoff 7 runs). torch and LightGBM run in separate processes.

Usage
    .venv/bin/python experiments/24_ensemble/ensemble.py \\
        --members lgb_mf30,lgb_pop,xgb_pop,lgb_mf20,mlp_pop --out submissions/24_ensemble.csv
"""
import argparse
import csv
import os
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "data" / "cache"
BASE = CACHE / "13_base_c7_9c47a117c762.npz"
MF = {"mf13": CACHE / "13_mf_c7_bb1fcb0a9c89.npz", "mf15": CACHE / "15_mf_c7_46f30c940a6a.npz",
      "mf20": CACHE / "20_mf_c7_c6b710c438b5.npz"}
ALL_MEMBERS = ["lgb_mf13", "lgb_mf15", "lgb_mf30", "lgb_pop", "xgb_pop", "lgb_mf20", "mlp_pop",
               "lgb_seq", "lgb_popseq", "xgb_popseq", "mlp_popseq", "lgb_full", "xgb_full", "mlp_full", "lgbrank_full",
               "choice_full", "lgbshallow_full", "lgbchoice_full"]
SEQ_NAMES = ["seq_logit", "seq_item_bias", "seq_user_bias", "seq_dot",
             "seq_minus_max", "seq_rank", "seq_share", "seq_lse"]
RANK_GROUP = int(os.environ.get("RANK_GROUP", 1000))
RANK_TREES = int(os.environ.get("RANK_TREES", 0))     # > 0: fixed lambdarank tree count, no early-stopping pass  # events per lambdarank group (LightGBM caps a query at 10,000 rows)
FIRST_DAY, ES_DAY = 2, 6                  # LightGBM rows: fit days 2-6; early stopping on day 6
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", eta=0.05,
                  max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, seed=0, nthread=int(os.environ.get("NTHREADS", 0)),
                  device=os.environ.get("XGB_DEVICE", "cpu"))     # XGB_DEVICE=cuda (2026-11-01): GPU on the vast.ai box

spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e13)                # numpy / scipy only at import time


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def pop_taste(d, fit, rows_ev, day_limit, n_items, hist, hlen):
    """(n_rows, 2): mean log-popularity of the user's known clicks, and candidate minus it.
    Popularity = clicks in the click log strictly before `day_limit` (a day index)."""
    keep = fit["clk_ts"] < e13.TRAIN_START + day_limit * e13.DAY
    lp = np.log1p(np.bincount(fit["clk_item"][keep], minlength=n_items)).astype(np.float32)
    hl = hlen[rows_ev].astype(np.float32)
    m = np.arange(hist.shape[1])[None, :] < hl[:, None]
    hp = np.where(hl > 0, np.where(m, lp[hist[rows_ev]], 0).sum(1) / np.maximum(hl, 1), np.nan)
    cp = lp[d["item"].reshape(-1, e13.EV_SIZE)[rows_ev]]
    return np.stack([np.repeat(hp, e13.EV_SIZE), cp.ravel() - np.repeat(hp, e13.EV_SIZE)], 1)


def score_group(fit_rows, caches, prefix):
    """8 features from alternative-MF caches (averaged): 4 scores + event-relative versions."""
    zs = [np.load(str(c)) for c in caches]
    Sf = np.mean([z["mf_fit"] for z in zs], 0)[fit_rows]
    St = np.mean([z["mf_test"] for z in zs], 0)
    if np.isnan(Sf).any() or np.isnan(St).any():
        raise ValueError(f"{prefix}: scores missing for some rows")
    if Sf.shape[1] != 4:      # 2026-11-03: raw multi-column view (lab/spectral_view.py), no event-relative columns
        return Sf.astype(np.float32), St.astype(np.float32), [f"{prefix}_{k}" for k in range(Sf.shape[1])]
    names = [f"{prefix}_{k}" for k in ("logit", "item_bias", "user_bias", "dot",
                                       "minus_max", "rank", "share", "lse")]
    return (np.concatenate([Sf, e13.relative_mf(Sf)], 1).astype(np.float32),
            np.concatenate([St, e13.relative_mf(St)], 1).astype(np.float32), names)


def load_all(seq_caches=(), extras=()):
    fit, test, eval_order, _, n_items = e13.load_base(str(BASE))
    mf = {k: np.load(str(v)) for k, v in MF.items()}
    fit_mf = {k: z["mf_fit"] for k, z in mf.items()}
    test_mf = {k: z["mf_test"] for k, z in mf.items()}
    fit_mf["mf30"] = (fit_mf["mf13"] + fit_mf["mf15"]) / 2
    test_mf["mf30"] = (test_mf["mf13"] + test_mf["mf15"]) / 2
    if MF_CORE_EXTRA:                       # each cache counts like one of 13's / 15's 15-seed sets
        ex = [np.load(str(p_)) for p_ in MF_CORE_EXTRA]
        fit_mf["mf30"] = (fit_mf["mf13"] + fit_mf["mf15"] + sum(z["mf_fit"] for z in ex)) / (2 + len(ex))
        test_mf["mf30"] = (test_mf["mf13"] + test_mf["mf15"] + sum(z["mf_test"] for z in ex)) / (2 + len(ex))
    row_day = np.repeat(fit["day"], e13.EV_SIZE)
    keep = np.flatnonzero(row_day >= FIRST_DAY)
    test_rows = np.arange(len(test["item"]))
    Xf, Xt, names = {}, {}, None
    for k in ("mf13", "mf15", "mf30", "mf20"):
        rows, Xf[k], names = e13.feature_matrix(fit, fit_mf[k], keep)
        _, Xt[k], _ = e13.feature_matrix(test, test_mf[k], test_rows)
    y, d = fit["label"][rows], row_day[rows]
    # popularity taste: fit rows as of their own day, evaluation rows as of the cutoff
    hist_f, hlen_f = e13.long_history(fit, fit, e13.HIST_LONG)
    hist_t, hlen_t = e13.long_history(test, fit, e13.HIST_LONG)
    pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))
    POPf = np.zeros((len(rows), 2), np.float32)
    for day in range(FIRST_DAY, e13.OFFICIAL_CUTOFF):
        e = np.flatnonzero(fit["day"] == day)
        POPf[pos[(e[:, None] * e13.EV_SIZE + np.arange(e13.EV_SIZE)).ravel()]] = \
            pop_taste(fit, fit, e, day, n_items, hist_f, hlen_f)
    POPt = pop_taste(test, fit, np.arange(len(test["user"])), e13.OFFICIAL_CUTOFF, n_items,
                     hist_t, hlen_t).astype(np.float32)
    Xf["pop"] = np.concatenate([Xf["mf30"], POPf], 1)
    Xt["pop"] = np.concatenate([Xt["mf30"], POPt], 1)
    names_pop = names + ["hist_pop", "cand_pop_minus_hist"]
    nm = {"base": names, "pop": names_pop}
    if seq_caches:
        zs = [np.load(str(c)) for c in seq_caches]
        Sf = np.mean([z["mf_fit"] for z in zs], 0)[rows]
        St = np.mean([z["mf_test"] for z in zs], 0)
        if np.isnan(Sf).any() or np.isnan(St).any():
            raise ValueError("sequence scores missing for some rows")
        SEQf = np.concatenate([Sf, e13.relative_mf(Sf)], 1).astype(np.float32)
        SEQt = np.concatenate([St, e13.relative_mf(St)], 1).astype(np.float32)
        Xf["seq"] = np.concatenate([Xf["mf30"], SEQf], 1)
        Xt["seq"] = np.concatenate([Xt["mf30"], SEQt], 1)
        Xf["popseq"] = np.concatenate([Xf["pop"], SEQf], 1)
        Xt["popseq"] = np.concatenate([Xt["pop"], SEQt], 1)
        nm["seq"] = names + SEQ_NAMES
        nm["popseq"] = names_pop + SEQ_NAMES
        log(f"  sequence features from {len(zs)} cache(s): " + ", ".join(Path(c).name for c in seq_caches))
        full_f, full_t, full_n = [Xf["popseq"]], [Xt["popseq"]], list(nm["popseq"])
        for name, caches in extras:
            gf, gt, gn = score_group(rows, caches, name)
            full_f.append(gf); full_t.append(gt); full_n += gn
            log(f"  extra group {name} from {len(caches)} cache(s): " + ", ".join(Path(c).name for c in caches))
        Xf["full"] = np.concatenate(full_f, 1)
        Xt["full"] = np.concatenate(full_t, 1)
        nm["full"] = full_n
    return fit, test, eval_order, Xf, Xt, nm, y, d


def gauss_rank(train_X, apply_X):
    """Per feature: normal scores by rank among the TRAINING rows; NaN -> 0 plus a flag."""
    outs = []
    for Xm in (train_X, apply_X):
        outs.append(np.zeros(Xm.shape, np.float32))
    flags_tr, flags_ap = [], []
    for j in range(train_X.shape[1]):
        ref_col = train_X[:, j].astype(np.float64)
        ref = np.sort(ref_col[~np.isnan(ref_col)])
        for Xm, out, flags in ((train_X, outs[0], flags_tr), (apply_X, outs[1], flags_ap)):
            col = Xm[:, j].astype(np.float64)
            nan = np.isnan(col)
            q = (np.searchsorted(ref, col, side="left") + np.searchsorted(ref, col, side="right")) / 2
            out[:, j] = np.where(nan, 0.0, ndtri(np.clip((q + 0.5) / (len(ref) + 1), 1e-4, 1 - 1e-4)))
            flags.append(nan.astype(np.float32))
    has_nan = [j for j in range(train_X.shape[1]) if flags_tr[j].any() or flags_ap[j].any()]
    tr = np.concatenate([outs[0]] + [flags_tr[j][:, None] for j in has_nan], 1)
    ap = np.concatenate([outs[1]] + [flags_ap[j][:, None] for j in has_nan], 1)
    return tr, ap


MLP_LR, MLP_SEEDS = 1e-3, 3                # defaults reproduce the earlier submissions
MF_CORE_EXTRA = []                          # --mf-core-extra: more MF caches averaged into the core "mf30"
MODEL_SEED = 0                              # --model-seed: LightGBM / XGBoost seed, MLP seeds start at 1000x it


def mlp_stage(out_path, feat, seq_caches, extras):
    """torch only (own process): MLP on the feature set `feat`; writes test predictions."""
    import torch
    torch.set_num_threads(8)
    _, _, _, Xf, Xt, _, y, d = load_all(seq_caches, extras)
    Ztr, Zte = gauss_rank(Xf[feat], Xt[feat])
    Zt, yt = torch.from_numpy(Ztr), torch.from_numpy(y.astype(np.float32))
    Ze = torch.from_numpy(Zte)
    es_tr, es_va, full_tr = np.flatnonzero(d < ES_DAY), np.flatnonzero(d == ES_DAY), np.arange(len(y))

    def net():
        return torch.nn.Sequential(
            torch.nn.Linear(Ztr.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def predict(m, Z):
        m.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(m(Z[s:s + 65536])).squeeze(1).numpy()
                                   for s in range(0, len(Z), 65536)])

    def fit_epochs(m, idx, epochs, r, opt, val=None):
        best, best_ep, bad = -1.0, 0, 0
        for epoch in range(epochs):
            m.train()
            perm = r.permutation(idx)
            for s in range(0, len(perm), 4096):
                b = torch.from_numpy(perm[s:s + 4096])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Zt[b]).squeeze(1), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            if val is not None:
                a = e13.roc_auc(predict(m, Zt[torch.from_numpy(val)]), y[val])
                if a > best:
                    best, best_ep, bad = a, epoch + 1, 0
                else:
                    bad += 1
                    if bad >= 2:
                        break
        return best, best_ep

    preds = []
    for seed in range(1000 * MODEL_SEED, 1000 * MODEL_SEED + MLP_SEEDS):
        torch.manual_seed(seed); r = np.random.default_rng(seed)
        m = net(); opt = torch.optim.Adam(m.parameters(), lr=MLP_LR, weight_decay=1e-6)
        best, n_ep = fit_epochs(m, es_tr, 30, r, opt, val=es_va)
        torch.manual_seed(seed); r = np.random.default_rng(seed)
        m = net(); opt = torch.optim.Adam(m.parameters(), lr=MLP_LR, weight_decay=1e-6)
        fit_epochs(m, full_tr, n_ep, r, opt)
        preds.append(predict(m, Ze))
        log(f"  mlp seed {seed}: {n_ep} epochs by early stopping on day {ES_DAY} "
            f"(day-{ES_DAY} AUC {best:.5f}), refit on days {FIRST_DAY}-{ES_DAY}")
    np.save(out_path, np.mean(preds, 0))


CHOICE_SEEDS = int(os.environ.get("CHOICE_SEEDS", 3))     # env override (2026-11-01: GPU builds)


def choice_stage(out_path, seq_caches, extras):
    """torch only (own process): neural choice model (lab/choice_model.py) on the full set; writes test predictions.
    One softmax per event over its 10 candidates + "no click": candidate scores from an MLP (256-128) on the Gaussian-rank
    features, the no-click score from an MLP (64) on the event-constant columns. Epochs by early stopping on day 6 (fit
    days 2-5), then refit on days 2-6 with that many epochs; CHOICE_SEEDS seeds averaged."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)
    dev = torch.device(os.environ.get("CHOICE_DEVICE", "cpu"))   # CHOICE_DEVICE=cuda (2026-11-01): GPU on the vast.ai box
    _, _, _, Xf, Xt, _, y, d = load_all(seq_caches, extras)
    Ztr, Zte = gauss_rank(Xf["full"], Xt["full"])
    del Xf, Xt
    E = e13.EV_SIZE
    Ef = torch.from_numpy(Ztr.reshape(-1, E, Ztr.shape[1])); Ee = torch.from_numpy(Zte.reshape(-1, E, Zte.shape[1]))
    smp = np.random.default_rng(0).choice(len(Ef), 20000, replace=False)
    const = torch.from_numpy(np.all(np.ptp(Ztr.reshape(-1, E, Ztr.shape[1])[smp], axis=1) < 1e-6, axis=0))
    lab = y.reshape(-1, E)
    target = torch.from_numpy(np.where(lab.sum(1) > 0, lab.argmax(1), E).astype(np.int64))
    Ef, Ee, const, target = Ef.to(dev), Ee.to(dev), const.to(dev), target.to(dev)
    d_ev = d.reshape(-1, E)[:, 0]
    es_tr, es_va, full_tr = np.flatnonzero(d_ev < ES_DAY), np.flatnonzero(d_ev == ES_DAY), np.arange(len(d_ev))
    log(f"  choice: {Ztr.shape[1]} inputs, {int(const.sum())} event-constant; {len(full_tr):,} events")

    def mlp(i, hs):
        layers = []
        for k in hs:
            layers += [torch.nn.Linear(i, k), torch.nn.ReLU(), torch.nn.Dropout(float(os.environ.get("CHOICE_DROP", 0.1)))]; i = k
        return torch.nn.Sequential(*layers, torch.nn.Linear(i, 1))

    class Choice(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cand = mlp(Ztr.shape[1], [int(v) for v in os.environ.get("CHOICE_HIDDEN", "256,128").split(",")]); self.none = mlp(int(const.sum()), [64])
        def forward(self, X3):
            return torch.cat([self.cand(X3).squeeze(2), self.none(X3[:, 0][:, const])], 1)

    def predict(m, X3, evs):
        m.eval()
        with torch.no_grad():
            out = [torch.softmax(m(X3[torch.from_numpy(evs[s:s + 8192]).to(dev)]), 1)[:, :E].cpu().numpy()
                   for s in range(0, len(evs), 8192)]
        m.train()
        return np.concatenate(out).ravel()

    def fit_epochs(m, opt, idx, epochs, r, val=None):
        best, best_ep, bad = -1.0, 0, 0
        for epoch in range(epochs):
            perm = r.permutation(idx)
            for s in range(0, len(perm), 512):
                b = torch.from_numpy(perm[s:s + 512]).to(dev)
                loss = F.cross_entropy(m(Ef[b]), target[b])
                opt.zero_grad(); loss.backward(); opt.step()
            if val is not None:
                a_ = e13.roc_auc(predict(m, Ef, val), lab[val].ravel())
                if a_ > best:
                    best, best_ep, bad = a_, epoch + 1, 0
                else:
                    bad += 1
                    if bad >= 2:
                        break
        return best, best_ep

    preds = []
    for seed in range(1000 * MODEL_SEED, 1000 * MODEL_SEED + CHOICE_SEEDS):
        torch.manual_seed(seed); r = np.random.default_rng(seed)
        m = Choice().to(dev); opt = torch.optim.Adam(m.parameters(), lr=float(os.environ.get("CHOICE_LR", 1e-3)))
        best, n_ep = fit_epochs(m, opt, es_tr, int(os.environ.get("CHOICE_EPOCHS", 12)), r, val=es_va)
        torch.manual_seed(seed); r = np.random.default_rng(seed)
        m = Choice().to(dev); opt = torch.optim.Adam(m.parameters(), lr=float(os.environ.get("CHOICE_LR", 1e-3)))
        fit_epochs(m, opt, full_tr, n_ep, r)
        preds.append(predict(m, Ee, np.arange(len(Ee))))
        log(f"  choice seed {seed}: {n_ep} epochs by early stopping on day {ES_DAY} (day-{ES_DAY} AUC {best:.5f}), "
            f"refit on days {FIRST_DAY}-{ES_DAY}")
    np.save(out_path, np.mean(preds, 0))


def tree_member(kind, X, Xt, names, y, d, lr=0.05):
    """Trees by early stopping on day 6 (fit 2-5), refit on 2-6 with the count scaled by 5/4."""
    es_tr, es_va = d < ES_DAY, d == ES_DAY
    scale = (ES_DAY - FIRST_DAY + 1) / (ES_DAY - FIRST_DAY)
    if kind in ("lgb", "lgbshallow"):
        import lightgbm as lgb
        p = {**e13.LGB_PARAMS, "learning_rate": lr, "seed": MODEL_SEED}
        if os.environ.get("NTHREADS"):    # NTHREADS=n (environment, 2026-11-01): cap threads for side-by-side jobs
            p["num_threads"] = int(os.environ["NTHREADS"])
        if kind == "lgbshallow":    # 2026-10-31 diversity member (lab/diversity_screen.sh W2): shallow, regularised trees
            p.update(num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.3)
        m = lgb.train(p, lgb.Dataset(X[es_tr], y[es_tr], feature_name=names), 8000,
                      valid_sets=[lgb.Dataset(X[es_va], y[es_va])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        best, n = m.best_score["valid_0"]["auc"], max(1, round(m.best_iteration * scale))
        m = lgb.train(p, lgb.Dataset(X, y, feature_name=names), n)
        return m.predict(Xt), best, n
    if kind == "lgbrank":
        # lambdarank over random groups of RANK_GROUP events (lab/global_rank2.py): the groups hold cross-event pairs, so
        # the objective is a surrogate of the global AUC. Rows come 10 per event, in event order.
        import lightgbm as lgb
        assert len(y) % e13.EV_SIZE == 0 and y.reshape(-1, e13.EV_SIZE).sum(1).max() <= 1
        p = {**e13.LGB_PARAMS, "learning_rate": lr, "seed": MODEL_SEED, "objective": "lambdarank", "metric": "auc",
             "lambdarank_truncation_level": 10000, "label_gain": [0, 1]}
        def grouped(mask):
            idx = np.flatnonzero(mask); ev = idx // e13.EV_SIZE
            uev = np.unique(ev); np.random.default_rng(MODEL_SEED).shuffle(uev)
            rank_of = np.empty(uev.max() + 1, np.int64); rank_of[uev] = np.arange(len(uev))
            key = rank_of[ev]; o = np.argsort(key, kind="stable")
            return idx[o], np.bincount(key[o] // RANK_GROUP)
        i_all, g_all = grouped(np.ones(len(y), bool))
        if RANK_TREES:
            # tree count fixed from earlier runs (36's G1000 build: 1790 trees by early stopping, 2238 after the refit
            # scaling): skips the early-stopping pass, which costs as much as the refit
            log(f"  lgbrank: {RANK_TREES} trees fixed (RANK_TREES), no early stopping; fitting on days {FIRST_DAY}-{ES_DAY}")
            m = lgb.train(p, lgb.Dataset(X[i_all], y[i_all], group=g_all, feature_name=names), RANK_TREES,
                          callbacks=[lgb.log_evaluation(250)])
            return m.predict(Xt), float("nan"), RANK_TREES
        i_tr, g_tr = grouped(es_tr)
        m = lgb.train(p, lgb.Dataset(X[i_tr], y[i_tr], group=g_tr, feature_name=names), 8000,
                      valid_sets=[lgb.Dataset(X[es_va], y[es_va], group=[int(es_va.sum())])],
                      callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)])
        log(f"  lgbrank: early stopping at {m.best_iteration} trees; refitting on days {FIRST_DAY}-{ES_DAY}")
        best, n = m.best_score["valid_0"]["auc"], max(1, round(m.best_iteration * scale))
        m = lgb.train(p, lgb.Dataset(X[i_all], y[i_all], group=g_all, feature_name=names), n)
        return m.predict(Xt), best, n
    if kind == "lgbchoice":
        # 2026-11-03 boosted choice model (lab/lgb_choice.py): conditional logit over the event's 10 candidates with the
        # no-click option at score 0; output p_i = exp(f_i) / (1 + sum_j exp(f_j)), comparable across events.
        import lightgbm as lgb
        from scipy.stats import rankdata
        assert len(y) % e13.EV_SIZE == 0 and len(Xt) % e13.EV_SIZE == 0
        def choice_p(f):
            e_ = f.reshape(-1, e13.EV_SIZE); m_ = np.maximum(e_.max(1, keepdims=True), 0); ex = np.exp(e_ - m_)
            return (ex / (np.exp(-m_) + ex.sum(1, keepdims=True))).reshape(-1)
        def auc(s, y_):
            r = rankdata(s); n1 = y_.sum()
            return float((r[y_ == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y_) - n1)))
        s_ = y.sum() / (len(y) / e13.EV_SIZE); f0 = float(np.log(s_ / e13.EV_SIZE / (1 - s_)))
        def fobj_for(yy):
            return lambda preds, data: (choice_p(preds) - yy, np.maximum(choice_p(preds) * (1 - choice_p(preds)), 1e-9))
        p = {**e13.LGB_PARAMS, "learning_rate": lr, "seed": MODEL_SEED, "metric": "None"}
        p.pop("objective", None)
        if os.environ.get("NTHREADS"):
            p["num_threads"] = int(os.environ["NTHREADS"])
        y_tr, y_va = y[es_tr].astype(np.float64), y[es_va].astype(np.float64)
        m = lgb.train({**p, "objective": fobj_for(y_tr)},
                      lgb.Dataset(X[es_tr], y_tr, init_score=np.full(len(y_tr), f0), feature_name=names), 8000,
                      valid_sets=[lgb.Dataset(X[es_va], y_va, init_score=np.full(len(y_va), f0))],
                      feval=lambda preds, data: ("auc_p", auc(choice_p(preds), y_va), True),
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        best, n = m.best_score["valid_0"]["auc_p"], max(1, round(m.best_iteration * scale))
        y_all_ = y.astype(np.float64)
        m = lgb.train({**p, "objective": fobj_for(y_all_)},
                      lgb.Dataset(X, y_all_, init_score=np.full(len(y_all_), f0), feature_name=names), n)
        return choice_p(m.predict(Xt, raw_score=True) + f0), best, n
    import xgboost as xgb
    dva = xgb.DMatrix(X[es_va], y[es_va])
    m = xgb.train({**XGB_PARAMS, "seed": MODEL_SEED}, xgb.DMatrix(X[es_tr], y[es_tr]), 4000, evals=[(dva, "v")],
                  early_stopping_rounds=100, verbose_eval=False)
    best, n = m.best_score, max(1, round((m.best_iteration + 1) * scale))
    m = xgb.train({**XGB_PARAMS, "seed": MODEL_SEED}, xgb.DMatrix(X, y), n)
    return m.predict(xgb.DMatrix(Xt)), best, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="lgb_mf30,lgb_pop,xgb_pop,lgb_mf20,mlp_pop")
    ap.add_argument("--weights", help="comma-separated rank weights (default: equal)")
    ap.add_argument("--seq-caches", default="",
                    help="comma-separated cutoff-7 caches of 18's --pure-seq runs (averaged)")
    ap.add_argument("--extra", action="append", default=[],
                    help="name=cache[,cache...]: a further alternative MF as 8 features (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", help=argparse.SUPPRESS)
    ap.add_argument("--mlp-out", help=argparse.SUPPRESS)
    ap.add_argument("--mlp-feat", help=argparse.SUPPRESS)
    ap.add_argument("--mlp-lr", type=float, default=1e-3, help="MLP learning rate (sweep: 3e-4 best)")
    ap.add_argument("--mlp-seeds", type=int, default=3, help="MLP seeds averaged")
    ap.add_argument("--mf-core-extra", default="",
                    help="comma-separated MF caches averaged into the core MF with 13's and 15's (empty = MF30)")
    ap.add_argument("--model-seed", type=int, default=0,
                    help="seed of LightGBM / XGBoost and offset of the MLP seeds (0 = earlier submissions)")
    a = ap.parse_args()
    global MLP_LR, MLP_SEEDS, MODEL_SEED, MF_CORE_EXTRA
    MLP_LR, MLP_SEEDS, MODEL_SEED = a.mlp_lr, a.mlp_seeds, a.model_seed
    MF_CORE_EXTRA = [p_ for p_ in a.mf_core_extra.split(",") if p_]
    seq_caches = [Path(c) for c in a.seq_caches.split(",") if c]
    extras = []
    for spec_ in a.extra:
        name, _, caches = spec_.partition("=")
        if not name or not caches:
            ap.error(f"--extra needs name=cache[,cache...], got {spec_!r}")
        extras.append((name, [Path(c) for c in caches.split(",") if c]))
    if a.stage == "mlp":
        mlp_stage(a.mlp_out, a.mlp_feat, seq_caches, extras)
        return
    if a.stage == "choice":
        choice_stage(a.mlp_out, seq_caches, extras)
        return
    members = a.members.split(",")
    unknown = [m for m in members if m not in ALL_MEMBERS]
    if unknown:
        ap.error(f"unknown members {unknown}; choose from {ALL_MEMBERS}")
    if any("seq" in m or "full" in m for m in members) and not seq_caches:
        ap.error("sequence and full members need --seq-caches")
    w = np.ones(len(members)) if not a.weights else np.array([float(x) for x in a.weights.split(",")])
    if len(w) != len(members):
        ap.error("--weights must have one value per member")

    preds = {}
    for mname, feat in (("mlp_pop", "pop"), ("mlp_popseq", "popseq"), ("mlp_full", "full")):
        if mname in members:              # torch first, in its own process
            mlp_file = CACHE / f"24_{mname}_test.npy"
            log(f"{mname}: training in a separate process ...")
            cmd = [sys.executable, __file__, "--stage", "mlp", "--mlp-out", str(mlp_file),
                   "--mlp-feat", feat, "--seq-caches", a.seq_caches, "--out", a.out]
            for spec_ in a.extra:
                cmd += ["--extra", spec_]
            cmd += ["--mlp-lr", str(a.mlp_lr), "--mlp-seeds", str(a.mlp_seeds), "--model-seed", str(a.model_seed), "--mf-core-extra", a.mf_core_extra]
            subprocess.run(cmd, check=True)
            preds[mname] = np.load(mlp_file)
    if "choice_full" in members:          # torch, own process (lab/choice_model.py)
        ch_file = CACHE / f"24_choice_full{os.environ.get('CHOICE_TAG', '')}_s{a.model_seed}_test.npy"   # CHOICE_TAG: variant
        log("choice_full: training in a separate process ...")
        cmd = [sys.executable, __file__, "--stage", "choice", "--mlp-out", str(ch_file), "--seq-caches", a.seq_caches,
               "--out", a.out, "--model-seed", str(a.model_seed), "--mf-core-extra", a.mf_core_extra]
        for spec_ in a.extra:
            cmd += ["--extra", spec_]
        subprocess.run(cmd, check=True)
        preds["choice_full"] = np.load(ch_file)
    fit, test, eval_order, Xf, Xt, nm, y, d = load_all(seq_caches, extras)
    spec_of = {"lgb_mf13": ("lgb", "mf13", "base", 0.05), "lgb_mf15": ("lgb", "mf15", "base", 0.05),
               "lgb_mf30": ("lgb", "mf30", "base", 0.05), "lgb_pop": ("lgb", "pop", "pop", 0.02),
               "xgb_pop": ("xgb", "pop", "pop", 0.05), "lgb_mf20": ("lgb", "mf20", "base", 0.05),
               "lgb_seq": ("lgb", "seq", "seq", 0.05), "lgb_popseq": ("lgb", "popseq", "popseq", 0.02),
               "xgb_popseq": ("xgb", "popseq", "popseq", 0.05),
               "lgb_full": ("lgb", "full", "full", 0.02), "xgb_full": ("xgb", "full", "full", 0.05),
               "lgbrank_full": ("lgbrank", "full", "full", 0.02), "lgbshallow_full": ("lgbshallow", "full", "full", 0.02),
               "lgbchoice_full": ("lgbchoice", "full", "full", 0.02)}
    for mname in members:
        if mname.startswith("mlp_") or mname == "choice_full":
            continue
        kind, fx, nk, lr = spec_of[mname]
        names_m = nm[nk]
        t0 = time.time()
        p, best, n = tree_member(kind, Xf[fx], Xt[fx], names_m, y, d, lr)
        preds[mname] = p
        log(f"  {mname}: {n} trees (day-{ES_DAY} AUC {best:.5f}, {time.time()-t0:.0f}s)")

    n = len(test["item"])
    for m in members:
        if len(preds[m]) != n or not np.all(np.isfinite(preds[m])):
            raise ValueError(f"{m}: wrong length or non-finite predictions")
    if os.environ.get("SAVE_RAW"):         # 2026-11-04: keep the members' raw probabilities for the conditional-logit stacker
        np.savez(os.environ["SAVE_RAW"], **{m: preds[m] for m in members})
    rk = {m: rankdata(preds[m]) for m in members}
    score = sum(wi * rk[m] / n for wi, m in zip(w, members)) / w.sum()
    if len(members) > 1:
        log("  rank correlations between members on the evaluation rows:")
        for i, m1 in enumerate(members):
            log("    " + m1 + ": " + ", ".join(f"{m2} {np.corrcoef(rk[m1], rk[m2])[0, 1]:.4f}"
                                              for m2 in members[i + 1:]))
    score = np.clip(score, 0.0, 1.0)
    by_rowid = dict(zip(test["rowid"].tolist(), score.tolist()))
    if len(by_rowid) != len(eval_order) or set(by_rowid) != set(eval_order.tolist()):
        raise ValueError("predicted rowids do not match evaluation.csv")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rowid", "predicted_probability"])
        for rid in eval_order.tolist():
            wr.writerow([rid, f"{by_rowid[rid]:.10g}"])
    log(f"done. wrote {a.out} ({len(members)} members: {', '.join(members)})")


if __name__ == "__main__":
    main()
