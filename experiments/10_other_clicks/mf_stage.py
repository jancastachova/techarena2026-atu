#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 10 (run by other_clicks.py as a separate process,
because torch and lightgbm crash when loaded into the same process on macOS).

Experiment 09's MF (user vector over all known clicks, click-log loss, 10 seeds) with one
change in the click-log examples: the context of a logged click is the user's OTHER clicks
(before and after it), not only the clicks before it. A history is an unordered set (order and
timing carry no signal, see RESULTS.md), so every other click is valid evidence for learning
the item vectors. This roughly doubles the context per example (mean 12.6 -> 23.5 clicks).

Time split: for the MF used for day d, the click log is cut to clicks before day d BEFORE the
contexts are built, so a context never contains a click on or after day d. Scoring is
unchanged: an event's user vector is still its user's clicks strictly before the event.

Chosen inside the fit period only (MF trained on days 0-3 + clicks before day 4, AUC on
fit day 4):
                                         1 seed   3 seeds
    09 (context = previous clicks)       0.752    0.772
    10 (context = other clicks)          0.779    0.791
    10, click-log weight 0.05/0.2/0.3/0.5: 0.773 / 0.779 / 0.776 / 0.772 (1 seed; 0.1 kept)
    09, l2 0.08 instead of 0.05:         0.755    0.770   (kept 0.05)

Usage (normally called by other_clicks.py):
    .venv/bin/python experiments/10_other_clicks/mf_stage.py --base <base.npz> --out <mf.npz> --cutoff 5
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

# experiment 09's MF stage, loaded by path (same file name as this one)
_spec = importlib.util.spec_from_file_location(
    "mf_stage_09", ROOT / "experiments" / "09_full_history" / "mf_stage.py")
m9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m9)
m5, m4 = m9.m5, m9.m4
from metrics import roc_auc           # noqa: E402  (src/ is on sys.path through m5)

EV_SIZE = m4.EV_SIZE
N_SEEDS = m9.N_SEEDS
HIST_LONG = m9.HIST_LONG
log = m5.log


def other_clicks_examples(fit, day, k):
    """Click-log examples from clicks before `day`: context = the user's other clicks before
    `day` (most recent k, the click itself excluded), target = the clicked item.
    Returns (ctx, ctx_len, target, ts) like experiment 05's click_examples()."""
    u, it, ts = fit["clk_user"], fit["clk_item"], fit["clk_ts"]
    keep = ts < m5.TRAIN_START + day * m5.DAY
    u, it, ts = u[keep], it[keep], ts[keep]
    o = np.lexsort((it, ts, u))
    u, it, ts = u[o], it[o], ts[o]
    bounds = np.r_[0, np.flatnonzero(np.diff(u)) + 1, len(u)]
    ctx = np.zeros((len(u), k), dtype=np.int64)
    ln = np.zeros(len(u), dtype=np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        items = it[a:b]
        for j in range(b - a):
            others = np.r_[items[:j], items[j + 1:]][-k:]
            ctx[a + j, :len(others)] = others
            ln[a + j] = len(others)
    if len(ts) and ts.max() >= m5.TRAIN_START + day * m5.DAY:
        raise AssertionError("click context contains a click on/after the scored day")
    has = ln > 0
    return ctx[has], ln[has], it[has], ts[has]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=int, required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)

    fit, test, n_users, n_items = m9.load_base(a.base)
    m9.check_against_event_history(fit, fit, "fit events")
    m9.check_against_event_history(test, fit, "test events")
    fit_l, test_l = m9.with_long_history(fit, fit, HIST_LONG), m9.with_long_history(test, fit, HIST_LONG)
    log(f"  history up to {HIST_LONG} clicks, {N_SEEDS} seeds, click context = other clicks")

    ev_day = fit["day"]
    mf_fit = np.full((len(ev_day) * EV_SIZE, len(m4.MF_FEATURES)), np.nan, dtype=np.float32)
    for d in range(m4.FIRST_MODEL_DAY, a.cutoff + 1):
        train_ev = np.flatnonzero(ev_day < d)
        clk_d = other_clicks_examples(fit, d, HIST_LONG)
        log(f"  clicks before day {d}: {len(clk_d[2]):,} examples, mean context {clk_d[1].mean():.1f}")
        if d < a.cutoff:
            target = np.flatnonzero(ev_day == d)
            log(f"  MF trained on days 0-{d-1} -> scoring fit day {d}")
            rows = (target[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
            mf_fit[rows] = m9.train_and_score(fit_l, train_ev, clk_d, n_users, n_items,
                                              lambda m: m4.score(m, fit_l, target), N_SEEDS)
            log(f"    AUC on day {d}: {roc_auc(mf_fit[rows, 0], fit['label'][rows]):.5f}")
        else:
            log(f"  MF trained on days 0-{d-1} -> scoring the test events")
            mf_test = m9.train_and_score(fit_l, train_ev, clk_d, n_users, n_items,
                                         lambda m: m4.score(m, test_l, np.arange(len(test["user"]))), N_SEEDS)
    np.savez(a.out, mf_fit=mf_fit, mf_test=mf_test)


if __name__ == "__main__":
    main()
