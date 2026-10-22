#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 09 (run by full_history.py as a separate process,
because torch and lightgbm crash when loaded into the same process on macOS).

Experiment 07's MF (history-only MF + click-log loss, averaged over N_SEEDS seeds), but each
event's user vector is the mean over ALL the user's logged clicks strictly before the event
(the HIST_LONG most recent), instead of the <= 20 clicks in `history_item_ids`. The click-log
examples use the same longer context. Time split unchanged: an MF used for day d only sees
impressions of days < d and clicks before day d.

HIST_LONG is chosen with --tune inside the fit period only (MF trained on days 0-3 + clicks
before day 4, AUC on fit day 4; 1 seed).

Usage (normally called by full_history.py):
    .venv/bin/python experiments/09_full_history/mf_stage.py --base <base.npz> --out <mf.npz> --cutoff 5
    .venv/bin/python experiments/09_full_history/mf_stage.py --base <base.npz> --tune 20,50,100
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

# experiment 05's MF stage (click log, model of experiment 04), loaded by path (same file name)
_spec = importlib.util.spec_from_file_location(
    "mf_stage_05", ROOT / "experiments" / "05_click_aug" / "mf_stage.py")
m5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m5)
m4 = m5.m4
from metrics import roc_auc           # noqa: E402  (src/ is on sys.path through m5)

EV_SIZE = m4.EV_SIZE
N_SEEDS = 10
HIST_LONG = 100                       # see --tune
log = m5.log


def load_base(path):
    fit, test, n_users, n_items = m5.load_base(path)
    z = np.load(path)
    fit["ts"], test["ts"] = z["tr_ts"], z["ev_ts"]
    return fit, test, n_users, n_items


def long_history(d, fit, k):
    """Per event of d: the user's logged clicks strictly before the event, newest first,
    at most k (0 = padding), and their count. The click log is in fit["clk_*"]."""
    u, it, ts = fit["clk_user"].astype(np.int64), fit["clk_item"], fit["clk_ts"].astype(np.int64)
    order = np.lexsort((it, ts, u))
    key = u[order] * 10_000_000_000 + ts[order]
    it = it[order]
    eu = d["user"].astype(np.int64)
    start = np.searchsorted(key, eu * 10_000_000_000, side="left")
    end = np.searchsorted(key, eu * 10_000_000_000 + d["ts"].astype(np.int64), side="left")
    hist = np.zeros((len(eu), k), dtype=np.int32)
    for j in range(k):
        idx = end - 1 - j
        ok = idx >= start
        hist[ok, j] = it[idx[ok]]
    return hist, np.minimum(end - start, k).astype(np.int8 if k < 128 else np.int16)


def check_against_event_history(d, fit, name):
    """The 20 most recent logged clicks must be the event's own history (as a multiset)."""
    h20, n20 = long_history(d, fit, 20)
    same = 0
    for e in range(len(n20)):
        a = np.sort(h20[e, :n20[e]])
        b = np.sort(d["hist"][e, :d["hlen"][e]])
        same += np.array_equal(a, b)
    rate = same / len(n20)
    log(f"  {name}: last 20 logged clicks == event history for {rate:.4%} of events")
    if rate < 0.99:
        raise AssertionError(f"{name}: click log and event histories disagree ({rate:.2%})")


def with_long_history(d, fit, k):
    d = dict(d)
    d["hist"], d["hlen"] = long_history(d, fit, k)
    return d


def train_and_score(fit, events, clicks, n_users, n_items, score_fn, n_seeds):
    out = None
    for seed in range(n_seeds):
        m5.SEED = seed
        s = score_fn(m5.train(fit, events, clicks, n_users, n_items, m5.AUX_WEIGHT))
        out = s if out is None else out + s
    m5.SEED = 0
    return out / n_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out")
    ap.add_argument("--cutoff", type=int)
    ap.add_argument("--tune", help="comma-separated HIST_LONG values (or 'event') to compare on fit day 4")
    a = ap.parse_args()
    torch.set_num_threads(8)

    fit, test, n_users, n_items = load_base(a.base)
    check_against_event_history(fit, fit, "fit events")
    check_against_event_history(test, fit, "test events")

    if a.tune:
        train_ev, target = np.flatnonzero(fit["day"] < 4), np.flatnonzero(fit["day"] == 4)
        rows = (target[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
        for v in a.tune.split(","):
            if v == "event":                       # experiment 05/07 behaviour
                m5.HIST_MAX, d = 20, fit
            else:
                m5.HIST_MAX, d = int(v), with_long_history(fit, fit, int(v))
            clicks = m5.clicks_before(m5.click_examples(fit), 4)
            s = train_and_score(d, train_ev, clicks, n_users, n_items, lambda m: m4.score(m, d, target), 1)
            log(f"=== history {v}: fit day-4 AUC {roc_auc(s[:, 0], fit['label'][rows]):.5f}")
        return

    m5.HIST_MAX = HIST_LONG
    clicks = m5.click_examples(fit)
    fit_l, test_l = with_long_history(fit, fit, HIST_LONG), with_long_history(test, fit, HIST_LONG)
    log(f"  {len(clicks[2]):,} logged clicks with context; history up to {HIST_LONG} clicks, "
        f"{N_SEEDS} seeds; mean history length {fit_l['hlen'].mean():.1f} (fit), "
        f"{test_l['hlen'].mean():.1f} (test) vs {fit['hlen'].mean():.1f} / {test['hlen'].mean():.1f} before")

    ev_day = fit["day"]
    mf_fit = np.full((len(ev_day) * EV_SIZE, len(m4.MF_FEATURES)), np.nan, dtype=np.float32)
    for d in range(m4.FIRST_MODEL_DAY, a.cutoff + 1):
        train_ev = np.flatnonzero(ev_day < d)
        clk_d = m5.clicks_before(clicks, d)
        if d < a.cutoff:
            target = np.flatnonzero(ev_day == d)
            log(f"  MF trained on days 0-{d-1} -> scoring fit day {d}")
            rows = (target[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
            mf_fit[rows] = train_and_score(fit_l, train_ev, clk_d, n_users, n_items,
                                           lambda m: m4.score(m, fit_l, target), N_SEEDS)
            log(f"    AUC on day {d}: {roc_auc(mf_fit[rows, 0], fit['label'][rows]):.5f}")
        else:
            log(f"  MF trained on days 0-{d-1} -> scoring the test events")
            mf_test = train_and_score(fit_l, train_ev, clk_d, n_users, n_items,
                                      lambda m: m4.score(m, test_l, np.arange(len(test["user"]))), N_SEEDS)
    np.savez(a.out, mf_fit=mf_fit, mf_test=mf_test)


if __name__ == "__main__":
    main()
