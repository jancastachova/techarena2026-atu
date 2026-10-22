#!/usr/bin/env python3
"""
TechArena CTR -- experiment 09: represent each user by ALL their known clicks, not just the
20 in `history_item_ids`.

Experiments 04-08 represent a user by the mean of the item vectors in the event's history,
which holds at most the 20 most recent clicks. About 60% of rows have a full 20-click history,
so older clicks are cut off. The order and timing of clicks carry no signal (see RESULTS.md),
so older clicks are as informative as recent ones. The click log of experiment 05 has every
click known before the cutoff, with timestamps; here an event's user vector uses all of the
user's logged clicks strictly before the event's timestamp (up to HIST_LONG, see mf_stage.py).
For evaluation events that is every pre-cutoff click of the user (a test event is the user's
first event at/after the cutoff, so no clicks lie between the cutoff and the event).

This file is experiment 05's pipeline (feature build with click log) plus the event timestamps,
which the cached features did not store. They are recorded by wrapping DailyState's
event_features(), which build() calls once per stored event, in storage order; build() then
checks that they line up with the stored events (see below).

Usage:
    .venv/bin/python experiments/09_full_history/full_history.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv
    .venv/bin/python experiments/09_full_history/full_history.py --cutoff 7 \
        --train data/train.csv --evaluation data/evaluation.csv --out submissions/09_full_history.csv
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "05_click_aug"))
import click_aug as e5                # noqa: E402  (experiment 05: build with click log; no torch)

t3, e2, h1 = e5.t3, e5.e2, e5.h1
_EVENT_TS = []

# Column order written by mf_stage.py (not imported: it loads torch).
MF_FEATURES = ["mf_logit", "mf_item_bias", "mf_user_bias", "mf_hist_dot"]


class TsState(t3.DailyState):
    """DailyState that records the timestamp of every featurised event, in call order."""

    def event_features(self, user, ts, *args, **kwargs):
        _EVENT_TS.append(ts)
        return super().event_features(user, ts, *args, **kwargs)


def build(train_path, eval_path, cutoff):
    """Experiment 05's build plus fit["ts"] and test["ts"] (event timestamps)."""
    _EVENT_TS.clear()
    fit, test, eval_order, n_users, n_items = e5.build(train_path, eval_path, cutoff)
    n_fit, n_test = len(fit["user"]), len(test["user"])
    if len(_EVENT_TS) != n_fit + n_test:
        raise AssertionError(f"{len(_EVENT_TS)} timestamps for {n_fit} + {n_test} events")
    ts = np.array(_EVENT_TS, dtype=np.int64)
    fit["ts"], test["ts"] = ts[:n_fit], ts[n_fit:]

    # the timestamps must line up with the stored events
    if not np.array_equal((fit["ts"] - h1.TRAIN_START) // h1.DAY, fit["day"].astype(np.int64)):
        raise AssertionError("fit timestamps do not match the stored days")
    test_day = np.minimum((test["ts"] - h1.TRAIN_START) // h1.DAY, 127)
    if not np.array_equal(test_day, test["day"].astype(np.int64)):
        raise AssertionError("test timestamps do not match the stored days")
    if np.any(np.diff(fit["ts"]) < 0):
        raise AssertionError("fit timestamps are not in time order")
    # every clicked fit candidate is in the click log at (user, item, event timestamp)
    lab = fit["label"].reshape(-1, e2.EV_SIZE)
    ev, pos = np.nonzero(lab)
    clicked = set(zip(fit["user"][ev].tolist(), fit["item"].reshape(-1, e2.EV_SIZE)[ev, pos].tolist(),
                      fit["ts"][ev].tolist()))
    logged = set(zip(fit["clk_user"].tolist(), fit["clk_item"].tolist(), fit["clk_ts"].tolist()))
    if not clicked <= logged:
        raise AssertionError(f"{len(clicked - logged)} clicked candidates missing from the click log")
    t3.log(f"event timestamps verified: {n_fit:,} fit + {n_test:,} test events, "
           f"{len(clicked):,} clicks matched in the click log")
    return fit, test, eval_order, n_users, n_items


if __name__ == "__main__":
    t3.DailyState = TsState
    t3.build = build
    t3.MF_STAGE = HERE / "mf_stage.py"
    t3.MF_TAG = "09"
    t3.MF_DEPS = [HERE.parent / "05_click_aug" / "mf_stage.py", HERE.parent / "04_history_mf" / "mf_stage.py"]
    e2.MF_FEATURES = MF_FEATURES
    t3.main()
