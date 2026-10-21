#!/usr/bin/env python3
"""
TechArena CTR -- experiment 02: matrix factorisation + LightGBM.

Needs the project environment (see requirements.txt):
    uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt

How it works:
  1. Base features: the leak-free "as-of" features of experiment 01 (repeat clicks,
     co-click similarity with the history, transitions, item/user stats ...),
     computed by streaming train.csv in time order, then evaluation.csv.
  2. Matrix factorisation scores from mf_model.py (run as a separate process, see
     there): rows of day d are scored by an MF trained on days < d only, the
     evaluation rows by an MF trained on all 7 days.
  3. LightGBM (binary objective; the leaderboard ranks by global AUC) on base +
     MF features, plus how each candidate compares with the other 9 of its event.
     Rows from days 2-6 are used (days 0-1 only feed the stats and the MFs).
  4. Validation (honest): LightGBM fit on days 2-4 with early stopping on day 5 to
     pick the number of trees, then refit on days 2-5 and report AUC on day 6.
     Final: refit on days 2-6 with that number of trees (+20% for the extra day),
     score evaluation.csv.

Base features and MF scores are cached in data/cache/ (keyed by a hash of the
code and inputs that produce them), so re-runs only redo what changed.

Usage:
    .venv/bin/python experiments/02_mf_lgbm/mf_lgbm.py \
    --train data/train.csv \
    --evaluation data/evaluation.csv \
    --out submissions/02_mf_lgbm.csv
"""
import argparse
import csv
import hashlib
import inspect
import os
import subprocess
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "01_history_recency"))
sys.path.insert(0, str(ROOT / "src"))
import history_recency as h1          # noqa: E402  (as-of feature builder of experiment 01)
from metrics import roc_auc           # noqa: E402

MF_SCRIPT = HERE / "mf_model.py"
HIST_MAX = 20
EV_SIZE = 10
FIRST_MODEL_DAY = 2                   # must match mf_model.FIRST_MODEL_DAY
SEED = 0

LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=SEED, deterministic=True, force_row_wise=True,
)

BASE_FEATURES = h1.CAND_FEATURES + h1.EVENT_FEATURES[:h1.NE_RAW]
MF_FEATURES = ["mf_logit", "mf_item_bias", "mf_user_dot", "mf_hist_dot"]   # order from mf_model.py
REL_FEATURES = ["mf_logit_minus_max", "mf_logit_rank", "mf_logit_share", "mf_event_lse",
                "sim_sum_minus_max", "sim_sum_rank", "ctr_minus_max"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def code_hash(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode() if isinstance(p, str) else p)
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- base features

class Split:
    """Columns of one split (train or eval); rows are grouped by event, EV_SIZE per event."""

    def __init__(self):
        self.cand = array("f")
        self.ev = array("f")
        self.user = array("i")       # per event
        self.item = array("i")       # per row
        self.hist = array("i")       # per event, HIST_MAX item codes, 0 = padding
        self.hlen = array("b")       # per event
        self.label = array("b")      # per row (train only)
        self.day = array("b")        # per event (train only)
        self.rowid = []              # per row (eval only)

    def add(self, cand, ev, user, items, h_items, enc_u, enc_i):
        if len(items) != EV_SIZE:
            raise ValueError(f"event with {len(items)} candidates, expected {EV_SIZE}")
        for r in cand:
            self.cand.extend(r)
        self.ev.extend(ev)
        self.user.append(enc_u(user))
        self.item.extend(enc_i(it) for it in items)
        codes = [enc_i(it) for it in h_items[-HIST_MAX:]]
        self.hist.extend(codes + [0] * (HIST_MAX - len(codes)))
        self.hlen.append(len(codes))

    def to_numpy(self):
        n_ev = len(self.user)
        out = dict(
            cand=np.frombuffer(self.cand, dtype=np.float32).reshape(n_ev * EV_SIZE, h1.NC),
            ev=np.frombuffer(self.ev, dtype=np.float32).reshape(n_ev, h1.NE_RAW),
            user=np.frombuffer(self.user, dtype=np.int32),
            item=np.frombuffer(self.item, dtype=np.int32),
            hist=np.frombuffer(self.hist, dtype=np.int32).reshape(n_ev, HIST_MAX),
            hlen=np.frombuffer(self.hlen, dtype=np.int8),
        )
        if len(self.label):
            out["label"] = np.frombuffer(self.label, dtype=np.int8)
            out["day"] = np.frombuffer(self.day, dtype=np.int8)
        if self.rowid:
            out["rowid"] = np.array(self.rowid)
        return {k: v.copy() for k, v in out.items()}


def build_base(train_path, eval_path):
    st = h1.State()
    users, items = {}, {}

    def enc_u(u):
        c = users.get(u)
        if c is None:
            c = users[u] = len(users)
        return c

    def enc_i(i):                    # item codes start at 1, 0 is the history padding
        c = items.get(i)
        if c is None:
            c = items[i] = len(items) + 1
        return c

    tr = Split()
    prev_ts = -1
    t0 = time.time()
    log("base features: streaming train.csv ...")
    for n_ev, (ix, rows) in enumerate(h1.iter_train_events(train_path)):
        user, ts, tb, dt, h_items, h_ts, its = h1.parse_event(rows, ix)
        if ts < prev_ts:
            raise ValueError("train.csv is not sorted by timestamp")
        prev_ts = ts
        day = (ts - h1.TRAIN_START) // h1.DAY
        if not 0 <= day <= 6:
            raise ValueError(f"unexpected training timestamp {ts}")
        ys = [int(r[ix["clicked"]]) for r in rows]
        if sum(ys) > 1:
            raise ValueError("event with more than one click")
        st.add_history(user, h_items, h_ts)
        cand, ev = st.event_features(user, ts, tb, dt, h_items, h_ts, its)
        tr.add(cand, ev, user, its, h_items, enc_u, enc_i)
        tr.label.extend(ys)
        tr.day.append(day)
        st.add_labels(user, its, ys)
        if (n_ev + 1) % 200_000 == 0:
            log(f"  {n_ev+1:,} events ({time.time()-t0:.0f}s)")

    log("base features: evaluation.csv in time order ...")
    with open(eval_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        ix = {c: i for i, c in enumerate(next(r))}
        eval_rows = list(r)
    events = defaultdict(list)
    for i, row in enumerate(eval_rows):
        events[row[ix["event_id"]]].append(i)
    order = sorted(events.values(), key=lambda idx: int(eval_rows[idx[0]][ix["timestamp"]]))
    ev_split = Split()
    for idx in order:
        rows = [eval_rows[i] for i in idx]
        user, ts, tb, dt, h_items, h_ts, its = h1.parse_event(rows, ix)
        st.add_history(user, h_items, h_ts)
        cand, ev = st.event_features(user, ts, tb, dt, h_items, h_ts, its)
        ev_split.add(cand, ev, user, its, h_items, enc_u, enc_i)
        ev_split.rowid.extend(r[ix["rowid"]] for r in rows)
    eval_order = np.array([row[ix["rowid"]] for row in eval_rows])
    log(f"base features done ({time.time()-t0:.0f}s): {len(tr.label):,} train rows, "
        f"{len(ev_split.rowid):,} eval rows, {len(users):,} users, {len(items):,} items")
    return tr.to_numpy(), ev_split.to_numpy(), eval_order, len(users), len(items) + 1


# --------------------------------------------------------------------------- LightGBM

def _grouped(x):
    return x.reshape(-1, EV_SIZE)


def _flat(cols):
    return np.stack([np.asarray(x).reshape(-1) for x in cols], 1).astype(np.float32)


def relative_base(base):
    """Base features compared with the other candidates of the event (REL_FEATURES[4:])."""
    s = _grouped(base[:, BASE_FEATURES.index("sim_sum")])
    c = _grouped(base[:, BASE_FEATURES.index("item_ctr_logit")])
    return _flat([s - s.max(1, keepdims=True), rankdata(-s, axis=1), c - c.max(1, keepdims=True)])


def relative_mf(mf):
    """MF score compared with the other candidates of the event (REL_FEATURES[:4])."""
    m = _grouped(mf[:, 0])
    m_max = m.max(1, keepdims=True)
    ex = np.exp(m - m_max)
    lse = m_max + np.log(ex.sum(1, keepdims=True))
    return _flat([m - m_max, rankdata(-m, axis=1), ex / ex.sum(1, keepdims=True),
                  np.broadcast_to(lse, m.shape)])


def feature_matrix(d, mf, rows, use_mf):
    """Feature matrix for the given rows (whole events only); returns (rows, X, names)."""
    ev_idx = np.unique(rows // EV_SIZE)
    full = (ev_idx[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
    if not np.array_equal(np.sort(rows), full):
        raise ValueError("feature_matrix needs complete events")
    base = np.concatenate([d["cand"][full], np.repeat(d["ev"][ev_idx], EV_SIZE, axis=0)], 1)
    if not use_mf:
        return full, np.concatenate([base, relative_base(base)], 1), BASE_FEATURES + REL_FEATURES[4:]
    m = mf[full]
    if np.isnan(m).any():
        raise ValueError("missing MF scores for some rows")
    X = np.concatenate([base, m, relative_mf(m), relative_base(base)], 1)
    return full, X, BASE_FEATURES + MF_FEATURES + REL_FEATURES


def lgb_fit(X, y, names, rounds, valid=None):
    ds = lgb.Dataset(X, y, feature_name=names, free_raw_data=False)
    kw = {}
    if valid is not None:
        kw = dict(valid_sets=[lgb.Dataset(valid[0], valid[1], reference=ds)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(LGB_PARAMS, ds, num_boost_round=rounds, **kw)


def validate(tr, mf_tr, use_mf):
    """Trees picked on day 5 (fit days 2-4), then refit on days 2-5 and tested on day 6."""
    row_day = np.repeat(tr["day"], EV_SIZE)

    def mat(mask):
        rows, X, names = feature_matrix(tr, mf_tr, np.flatnonzero(mask), use_mf)
        return X, tr["label"][rows], names

    Xa, ya, names = mat((row_day >= FIRST_MODEL_DAY) & (row_day <= 4))
    Xb, yb, _ = mat(row_day == 5)
    m = lgb_fit(Xa, ya, names, 3000, valid=(Xb, yb))
    n_trees = m.best_iteration
    m = lgb_fit(np.concatenate([Xa, Xb]), np.concatenate([ya, yb]), names, n_trees)
    del Xa, Xb
    Xc, yc, _ = mat(row_day == 6)
    return roc_auc(m.predict(Xc), yc), n_trees, m, names


# --------------------------------------------------------------------------- main

def run_mf(base_file, mf_file, tune=False):
    """Runs mf_model.py in its own process (torch and lightgbm cannot share one on macOS)."""
    cmd = [sys.executable, str(MF_SCRIPT), "--base", str(base_file)]
    if tune:
        subprocess.run(cmd + ["--tune"], check=True)
        return
    tmp = mf_file.with_name(mf_file.stem + ".partial.npz")
    subprocess.run(cmd + ["--out", str(tmp)], check=True)
    os.replace(tmp, mf_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--tune-mf", action="store_true", help="only report MF day-6 AUC per epoch")
    a = ap.parse_args()

    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    inputs = f"{a.train}|{Path(a.train).stat().st_size}|{a.evaluation}|{Path(a.evaluation).stat().st_size}"
    src_01 = (ROOT / "experiments" / "01_history_recency" / "history_recency.py").read_bytes()
    base_key = code_hash(src_01, inputs, inspect.getsource(Split), inspect.getsource(build_base),
                         str(HIST_MAX), str(EV_SIZE))
    mf_key = code_hash(base_key, MF_SCRIPT.read_bytes())

    base_file = cache / f"02_base_{base_key}.npz"
    if not base_file.exists():
        tr, ev, eval_order, n_users, n_items = build_base(a.train, a.evaluation)
        tmp = base_file.with_name(base_file.stem + ".partial.npz")
        np.savez(tmp, eval_order=eval_order, n_users=n_users, n_items=n_items,
                 **{f"tr_{k}": v for k, v in tr.items()}, **{f"ev_{k}": v for k, v in ev.items()})
        os.replace(tmp, base_file)
    log(f"loading base features {base_file.name}")
    z = np.load(base_file)
    tr = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
    ev = {k[3:]: z[k] for k in z.files if k.startswith("ev_")}
    eval_order = z["eval_order"]

    if a.tune_mf:
        run_mf(base_file, None, tune=True)
        return

    mf_file = cache / f"02_mf_{mf_key}.npz"
    if not mf_file.exists():
        log("matrix factorisation (one model per scored day, separate process) ...")
        run_mf(base_file, mf_file)
    log(f"loading MF scores {mf_file.name}")
    z = np.load(mf_file)
    mf_tr, mf_ev = z["mf_tr"], z["mf_ev"]

    row_day = np.repeat(tr["day"], EV_SIZE)
    day6 = np.flatnonzero(row_day == 6)
    log(f"day-6 AUC of the MF score alone: {roc_auc(mf_tr[day6, 0], tr['label'][day6]):.5f}")

    log("LightGBM validation without MF features ...")
    auc0, n0, _, _ = validate(tr, mf_tr, use_mf=False)
    log(f"  day-6 AUC base features only: {auc0:.5f} ({n0} trees)")
    log("LightGBM validation with MF features ...")
    auc1, n_trees, model, names = validate(tr, mf_tr, use_mf=True)
    log(f"VALIDATION AUC (day 6), base + MF: {auc1:.5f} ({n_trees} trees)")
    gain = model.feature_importance("gain")
    for k in np.argsort(-gain)[:15]:
        log(f"  {names[k]:22s} gain {gain[k] / gain.sum():.3f}")

    log("final model: days 2-6 ...")
    rows, X, names = feature_matrix(tr, mf_tr, np.flatnonzero(row_day >= FIRST_MODEL_DAY), use_mf=True)
    final = lgb_fit(X, tr["label"][rows], names, int(round(n_trees * 1.2)))
    del X
    rows_e, Xe, _ = feature_matrix(ev, mf_ev, np.arange(len(ev["item"])), use_mf=True)
    pred = final.predict(Xe)
    if not np.all(np.isfinite(pred)) or pred.min() < 0 or pred.max() > 1:
        raise ValueError("predictions outside [0, 1]")
    by_rowid = dict(zip(ev["rowid"][rows_e].tolist(), pred.tolist()))
    if len(by_rowid) != len(eval_order) or set(by_rowid) != set(eval_order.tolist()):
        raise ValueError("predicted rowids do not match evaluation.csv")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rowid", "predicted_probability"])
        for rid in eval_order.tolist():
            w.writerow([rid, f"{by_rowid[rid]:.6g}"])
    log(f"done. wrote {a.out}")


if __name__ == "__main__":
    main()
