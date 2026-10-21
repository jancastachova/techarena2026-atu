#!/usr/bin/env python3
"""
TechArena CTR -- experiment 03: experiment 02 (MF + LightGBM) with a validation that
follows the Phase 1 "History and time split" rules, and the same procedure for the
submission.

One procedure, run at a cutoff day C (days counted from 2025-01-20 00:00 UTC):
  - Fit part: all training events before C.
  - Test events: each user's FIRST event at or after C, with all 10 candidates; later
    events of those users are not used at all (this mirrors evaluation.csv).
      * validation:  C = 5 (2025-01-25), test events come from train.csv (days 5-6)
      * submission:  C = 7 (2025-01-27, the official cutoff), test events = evaluation.csv
  - Labels from the test events are never used for features: label statistics (item CTR,
    user click rate) are frozen at C. Inside the fit part they are updated once per day
    (each event sees the labels of earlier days), so training rows have stale-ish stats
    like the test rows. Click histories are inputs and are used as-of each event.
  - MF scores (mf_stage.py): fit day d from an MF trained on days < d; test events from
    an MF trained on all days < C.
  - LightGBM on fit days 2..C-1: the number of trees is picked by early stopping on day
    C-1 (fit on days 2..C-2), then the model is refit on days 2..C-1 with the trees scaled
    by the extra data. Nothing is tuned on the test events.

Validation also prints AUC per segment (new users, empty history, days after the cutoff)
and the submission run prints how common those segments are in evaluation.csv.

Usage:
    # validation (no submission written)
    .venv/bin/python experiments/03_timesplit/timesplit.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv
    # submission
    .venv/bin/python experiments/03_timesplit/timesplit.py --cutoff 7 \
        --train data/train.csv --evaluation data/evaluation.csv --out submissions/03_timesplit.csv
"""
import argparse
import csv
import inspect
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "01_history_recency"))
sys.path.insert(0, str(ROOT / "experiments" / "02_mf_lgbm"))
sys.path.insert(0, str(ROOT / "src"))
import history_recency as h1          # noqa: E402  (as-of feature builder)
import mf_lgbm as e2                  # noqa: E402  (feature matrix + LightGBM helpers)
from metrics import roc_auc           # noqa: E402

MF_STAGE = HERE / "mf_stage.py"
MF_TAG = "03"                         # cache-file prefix of the MF scores (later experiments override)
MF_DEPS = [ROOT / "experiments" / "02_mf_lgbm" / "mf_model.py"]   # code MF_STAGE depends on (cache key)
EV_SIZE = e2.EV_SIZE
FIRST_MODEL_DAY = e2.FIRST_MODEL_DAY
OFFICIAL_CUTOFF = 7


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class DailyState(h1.State):
    """Label statistics only change when flush() is called (at day boundaries in the fit
    part); after the cutoff nothing is flushed, so test events never see their labels."""

    def __init__(self):
        super().__init__()
        self.pending = []

    def add_labels(self, user, items, labels):
        self.pending.append((user, items, labels))

    def flush(self):
        for args in self.pending:
            h1.State.add_labels(self, *args)
        self.pending = []


def build(train_path, eval_path, cutoff):
    st = DailyState()
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

    fit, test = e2.Split(), e2.Split()
    test_users = set()
    n_skipped = 0
    prev_ts, cur_day = -1, 0
    t0 = time.time()
    log(f"features with cutoff day {cutoff}: streaming train.csv ...")
    for n_ev, (ix, rows) in enumerate(h1.iter_train_events(train_path)):
        user, ts, tb, dt, h_items, h_ts, its = h1.parse_event(rows, ix)
        if ts < prev_ts:
            raise ValueError("train.csv is not sorted by timestamp")
        prev_ts = ts
        day = (ts - h1.TRAIN_START) // h1.DAY
        if not 0 <= day <= 6:
            raise ValueError(f"unexpected training timestamp {ts}")
        if day != cur_day:
            st.flush()               # labels of finished fit days become visible
            cur_day = day
        ys = [int(r[ix["clicked"]]) for r in rows]
        if sum(ys) > 1:
            raise ValueError("event with more than one click")
        if day < cutoff:
            st.add_history(user, h_items, h_ts)
            cand, ev = st.event_features(user, ts, tb, dt, h_items, h_ts, its)
            fit.add(cand, ev, user, its, h_items, enc_u, enc_i)
            fit.label.extend(ys)
            fit.day.append(day)
            st.add_labels(user, its, ys)
        elif user not in test_users:  # first event of this user at/after the cutoff
            test_users.add(user)
            st.add_history(user, h_items, h_ts)
            cand, ev = st.event_features(user, ts, tb, dt, h_items, h_ts, its)
            test.add(cand, ev, user, its, h_items, enc_u, enc_i)
            test.label.extend(ys)
            test.day.append(day)
            test.rowid.extend(r[ix["rowid"]] for r in rows)
        else:
            n_skipped += 1            # later events of test users: not used at all
        if (n_ev + 1) % 200_000 == 0:
            log(f"  {n_ev+1:,} events ({time.time()-t0:.0f}s)")
    if st.pending and cutoff < OFFICIAL_CUTOFF:
        raise AssertionError("labels after the cutoff were queued")

    eval_order = np.array([], dtype=str)
    if cutoff == OFFICIAL_CUTOFF:
        st.flush()
        log("features: evaluation.csv in time order ...")
        with open(eval_path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            ix = {c: i for i, c in enumerate(next(r))}
            eval_rows = list(r)
        events = defaultdict(list)
        for i, row in enumerate(eval_rows):
            events[row[ix["event_id"]]].append(i)
        order = sorted(events.values(), key=lambda idx: int(eval_rows[idx[0]][ix["timestamp"]]))
        for idx in order:
            rows = [eval_rows[i] for i in idx]
            user, ts, tb, dt, h_items, h_ts, its = h1.parse_event(rows, ix)
            if user in test_users:
                raise ValueError("evaluation.csv has more than one event for a user")
            test_users.add(user)
            st.add_history(user, h_items, h_ts)
            cand, ev = st.event_features(user, ts, tb, dt, h_items, h_ts, its)
            test.add(cand, ev, user, its, h_items, enc_u, enc_i)
            test.day.append(min((ts - h1.TRAIN_START) // h1.DAY, 127))
            test.rowid.extend(r[ix["rowid"]] for r in rows)
        eval_order = np.array([row[ix["rowid"]] for row in eval_rows])

    log(f"features done ({time.time()-t0:.0f}s): {len(fit.user):,} fit events, "
        f"{len(test.user):,} test events, {n_skipped:,} later events of test users skipped")
    fit_np, test_np = fit.to_numpy(), test.to_numpy()
    if "day" not in test_np:
        test_np["day"] = np.frombuffer(test.day, dtype=np.int8).copy()
    return fit_np, test_np, eval_order, len(users), len(items) + 1


def segments(fit, test):
    """Per test row: segment name -> boolean mask."""
    fit_users = np.unique(fit["user"])
    ev_new = ~np.isin(test["user"], fit_users)
    ev_empty = test["hlen"] == 0
    rep = test["cand"][:, h1.CAND_FEATURES.index("rep_any")] > 0
    off = np.repeat(test["day"], EV_SIZE)
    segs = {
        "all": np.ones(len(rep), dtype=bool),
        "user has fit events": np.repeat(~ev_new, EV_SIZE),
        "new user (no fit events)": np.repeat(ev_new, EV_SIZE),
        "empty history": np.repeat(ev_empty, EV_SIZE),
        "candidate in history": rep,
        "candidate not in history": ~rep,
    }
    for d in np.unique(off):
        segs[f"test day {int(d)}"] = off == d
    return segs


def lgbm_procedure(fit, mf_fit, cutoff, use_mf):
    """Trees picked by early stopping on day C-1 (fit days 2..C-2), refit on days 2..C-1."""
    row_day = np.repeat(fit["day"], EV_SIZE)
    last = cutoff - 1

    def mat(mask):
        rows, X, names = e2.feature_matrix(fit, mf_fit, np.flatnonzero(mask), use_mf)
        return X, fit["label"][rows], names

    Xa, ya, names = mat((row_day >= FIRST_MODEL_DAY) & (row_day < last))
    Xb, yb, _ = mat(row_day == last)
    m = e2.lgb_fit(Xa, ya, names, 3000, valid=(Xb, yb))
    n_es_days, n_all_days = last - FIRST_MODEL_DAY, last - FIRST_MODEL_DAY + 1
    n_trees = max(1, int(round(m.best_iteration * n_all_days / n_es_days)))
    log(f"  trees: {m.best_iteration} by early stopping on day {last} -> {n_trees} for days "
        f"{FIRST_MODEL_DAY}-{last} (best day-{last} AUC {m.best_score['valid_0']['auc']:.5f})")
    model = e2.lgb_fit(np.concatenate([Xa, Xb]), np.concatenate([ya, yb]), names, n_trees)
    return model, names


def run_mf(base_file, mf_file, cutoff):
    tmp = mf_file.with_name(mf_file.stem + ".partial.npz")
    subprocess.run([sys.executable, str(MF_STAGE), "--base", str(base_file), "--out", str(tmp),
                    "--cutoff", str(cutoff)], check=True)
    os.replace(tmp, mf_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--cutoff", type=int, default=5,
                    help="cutoff day: 3-6 = validation inside train.csv, 7 = submission")
    ap.add_argument("--out", help="submission path (cutoff 7 only)")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--base-only", action="store_true", help="build/cache the base features, print the path, stop")
    a = ap.parse_args()
    if not 4 <= a.cutoff <= OFFICIAL_CUTOFF:
        ap.error("cutoff must be between 4 and 7 (LightGBM needs fit days 2..C-1, at least 2 days)")
    if not a.base_only and (a.cutoff == OFFICIAL_CUTOFF) != bool(a.out):
        ap.error("--out is required with --cutoff 7 and only allowed there")

    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    inputs = f"{a.train}|{Path(a.train).stat().st_size}|{a.evaluation}|{Path(a.evaluation).stat().st_size}"
    src_01 = (ROOT / "experiments" / "01_history_recency" / "history_recency.py").read_bytes()
    base_key = e2.code_hash(src_01, inputs, str(a.cutoff), inspect.getsource(e2.Split),
                            inspect.getsource(DailyState), inspect.getsource(build))
    mf_key = e2.code_hash(base_key, MF_STAGE.read_bytes(), *(p.read_bytes() for p in MF_DEPS))

    base_file = cache / f"03_base_c{a.cutoff}_{base_key}.npz"
    if not base_file.exists():
        fit, test, eval_order, n_users, n_items = build(a.train, a.evaluation, a.cutoff)
        tmp = base_file.with_name(base_file.stem + ".partial.npz")
        np.savez(tmp, eval_order=eval_order, n_users=n_users, n_items=n_items,
                 **{f"tr_{k}": v for k, v in fit.items()}, **{f"ev_{k}": v for k, v in test.items()})
        os.replace(tmp, base_file)
    if a.base_only:
        print(base_file)
        return
    log(f"loading base features {base_file.name}")
    z = np.load(base_file)
    fit = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
    test = {k[3:]: z[k] for k in z.files if k.startswith("ev_")}
    eval_order = z["eval_order"]

    mf_file = cache / f"{MF_TAG}_mf_c{a.cutoff}_{mf_key}.npz"
    if not mf_file.exists():
        log("matrix factorisation (separate process) ...")
        run_mf(base_file, mf_file, a.cutoff)
    log(f"loading MF scores {mf_file.name}")
    z = np.load(mf_file)
    mf_fit, mf_test = z["mf_fit"], z["mf_test"]

    test_rows = np.arange(len(test["item"]))
    segs = segments(fit, test)

    if a.cutoff < OFFICIAL_CUTOFF:
        y = test["label"]
        log(f"\nVALIDATION, cutoff day {a.cutoff}: {len(test['user']):,} test events "
            f"(first event per user at/after the cutoff), click rate {y.mean():.4f}")
        results = {"MF score alone": mf_test[:, 0]}
        for use_mf, name in ((False, "LightGBM, no MF"), (True, "LightGBM + MF")):
            log(f"fitting {name} ...")
            model, _ = lgbm_procedure(fit, mf_fit, a.cutoff, use_mf)
            _, X, _ = e2.feature_matrix(test, mf_test, test_rows, use_mf)
            results[name] = model.predict(X)
        header = f"  {'segment':28s} {'rows':>7s} " + " ".join(f"{k:>16s}" for k in results)
        log(header)
        for seg, mask in segs.items():
            if mask.sum() == 0 or y[mask].min() == y[mask].max():
                continue
            aucs = " ".join(f"{roc_auc(p[mask], y[mask]):16.5f}" for p in results.values())
            log(f"  {seg:28s} {mask.mean():7.1%} {aucs}")
        return

    log("SUBMISSION, official cutoff: segment shares in evaluation.csv")
    for seg, mask in segs.items():
        log(f"  {seg:28s} {mask.mean():7.1%}")
    model, _ = lgbm_procedure(fit, mf_fit, a.cutoff, use_mf=True)
    _, X, _ = e2.feature_matrix(test, mf_test, test_rows, use_mf=True)
    pred = model.predict(X)
    if not np.all(np.isfinite(pred)) or pred.min() < 0 or pred.max() > 1:
        raise ValueError("predictions outside [0, 1]")
    by_rowid = dict(zip(test["rowid"].tolist(), pred.tolist()))
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
