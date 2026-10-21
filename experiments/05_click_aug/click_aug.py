#!/usr/bin/env python3
"""
TechArena CTR -- experiment 05: experiment 04 + past clicks as extra MF training examples.

The history MF of experiment 04 still gains ~0.03-0.05 AUC per extra day of data, so it is
data-limited. Click histories hold every click before the cutoff (including the warm-up
period before 2025-01-20), about twice as many clicks as the labelled impressions.

This file is experiment 03's pipeline (experiments/03_timesplit/timesplit.py) with:
  - build(): experiment 03's feature build, plus a click log (user, item, time) of every
    click known before the cutoff: history clicks and labelled clicks of fit events, and
    history clicks of the test events (those are all before the cutoff: a test event is
    the user's first event at/after it). Later events of test users are still not used.
  - mf_stage.py here: experiment 04's MF plus a loss on the click log (see there).

Usage:
    .venv/bin/python experiments/05_click_aug/click_aug.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv
    .venv/bin/python experiments/05_click_aug/click_aug.py --cutoff 7 \
        --train data/train.csv --evaluation data/evaluation.csv --out submissions/05_click_aug.csv
"""
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "03_timesplit"))
import timesplit as t3                # noqa: E402
import mf_lgbm as e2                  # noqa: E402  (on sys.path through timesplit)
import history_recency as h1          # noqa: E402  (on sys.path through timesplit)

# Column order written by mf_stage.py (not imported: it loads torch, which must not share
# a process with lightgbm). A mismatch in count makes LightGBM fail loudly.
MF_FEATURES = ["mf_logit", "mf_item_bias", "mf_user_bias", "mf_hist_dot"]


def build(train_path, eval_path, cutoff):
    """Experiment 03's build plus fit["clk_user"/"clk_item"/"clk_ts"]: the click log."""
    st = t3.DailyState()
    users, items = {}, {}
    clicks = set()

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

    def log_history(user, h_items, h_ts):
        uc = enc_u(user)
        for it, t in zip(h_items, h_ts):
            clicks.add((uc, enc_i(it), t))

    fit, test = e2.Split(), e2.Split()
    test_users = set()
    n_skipped = 0
    prev_ts, cur_day = -1, 0
    t0 = time.time()
    t3.log(f"features with cutoff day {cutoff}: streaming train.csv ...")
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
            log_history(user, h_items, h_ts)
            for it, y in zip(its, ys):
                if y:
                    clicks.add((enc_u(user), enc_i(it), ts))
        elif user not in test_users:  # first event of this user at/after the cutoff
            test_users.add(user)
            st.add_history(user, h_items, h_ts)
            cand, ev = st.event_features(user, ts, tb, dt, h_items, h_ts, its)
            test.add(cand, ev, user, its, h_items, enc_u, enc_i)
            test.label.extend(ys)
            test.day.append(day)
            test.rowid.extend(r[ix["rowid"]] for r in rows)
            log_history(user, h_items, h_ts)
        else:
            n_skipped += 1            # later events of test users: not used at all
        if (n_ev + 1) % 200_000 == 0:
            t3.log(f"  {n_ev+1:,} events ({time.time()-t0:.0f}s)")
    if st.pending and cutoff < t3.OFFICIAL_CUTOFF:
        raise AssertionError("labels after the cutoff were queued")

    eval_order = np.array([], dtype=str)
    if cutoff == t3.OFFICIAL_CUTOFF:
        st.flush()
        t3.log("features: evaluation.csv in time order ...")
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
            log_history(user, h_items, h_ts)
        eval_order = np.array([row[ix["rowid"]] for row in eval_rows])

    cutoff_ts = h1.TRAIN_START + cutoff * h1.DAY
    clk = np.array(sorted(clicks), dtype=np.int64).reshape(-1, 3)
    if len(clk) and clk[:, 2].max() >= cutoff_ts:
        raise AssertionError("click log contains a click at/after the cutoff")
    t3.log(f"features done ({time.time()-t0:.0f}s): {len(fit.user):,} fit events, "
           f"{len(test.user):,} test events, {n_skipped:,} later events of test users skipped, "
           f"{len(clk):,} logged clicks before the cutoff")
    fit_np, test_np = fit.to_numpy(), test.to_numpy()
    if "day" not in test_np:
        test_np["day"] = np.frombuffer(test.day, dtype=np.int8).copy()
    fit_np["clk_user"], fit_np["clk_item"], fit_np["clk_ts"] = clk[:, 0], clk[:, 1], clk[:, 2]
    return fit_np, test_np, eval_order, len(users), len(items) + 1


if __name__ == "__main__":
    t3.build = build
    t3.MF_STAGE = HERE / "mf_stage.py"
    t3.MF_TAG = "05"
    t3.MF_DEPS = [HERE.parent / "04_history_mf" / "mf_stage.py"]
    e2.MF_FEATURES = MF_FEATURES
    t3.main()
