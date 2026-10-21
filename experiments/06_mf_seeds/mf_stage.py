#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 06 (run by mf_seeds.py as a separate process,
because torch and lightgbm crash when loaded into the same process on macOS).

Experiment 05's MF (history-only MF + click-log loss), trained N_SEEDS times with different
random seeds; the MF features are averaged over the seeds. Single runs vary a lot, so the
average is much more accurate. Time split unchanged: an MF used for day d only sees
impressions of days < d and clicks before day d.

Chosen inside the fit period only (MF trained on days 0-3 + clicks before day 4, AUC on
fit day 4; cutoff-5 base features):
    1 seed, l2 0.05 (= experiment 05)   0.736
    1 seed, l2 0.03 / 0.08 / 0.12       0.730 / 0.739 / 0.738
    1 seed, 8000 steps / dim 64         0.733 / 0.735
    3 seeds, l2 0.05 / 0.08             0.758 / 0.757
    5 seeds, l2 0.05 / 0.08             0.764 / 0.762

Usage (normally called by mf_seeds.py):
    .venv/bin/python experiments/06_mf_seeds/mf_stage.py --base <base.npz> --out <mf.npz> --cutoff 5
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

# experiment 05's MF stage, loaded by path (same file name as this one)
_spec = importlib.util.spec_from_file_location(
    "mf_stage_05", ROOT / "experiments" / "05_click_aug" / "mf_stage.py")
m5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m5)
m4 = m5.m4                            # experiment 04's MF stage (model + base settings)
from metrics import roc_auc           # noqa: E402  (src/ is on sys.path through m5)

N_SEEDS = 5
MF_L2 = 0.05
log = m5.log


def train_and_score(fit, events, clicks, n_users, n_items, d, rows_fn):
    """Average of the MF features over N_SEEDS seeds; rows_fn(model) scores one model."""
    m4.MF_L2 = MF_L2
    out = None
    for seed in range(N_SEEDS):
        m5.SEED = seed
        s = rows_fn(m5.train(fit, events, clicks, n_users, n_items, m5.AUX_WEIGHT))
        out = s if out is None else out + s
    return out / N_SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=int, required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)

    fit, test, n_users, n_items = m5.load_base(a.base)
    clicks = m5.click_examples(fit)
    ev_day = fit["day"]
    log(f"  {len(clicks[2]):,} logged clicks with context; {N_SEEDS} seeds, l2 {MF_L2}")

    mf_fit = np.full((len(ev_day) * m4.EV_SIZE, len(m4.MF_FEATURES)), np.nan, dtype=np.float32)
    for d in range(m4.FIRST_MODEL_DAY, a.cutoff + 1):
        train_ev = np.flatnonzero(ev_day < d)
        clk_d = m5.clicks_before(clicks, d)
        if d < a.cutoff:
            target = np.flatnonzero(ev_day == d)
            log(f"  MF trained on days 0-{d-1} -> scoring fit day {d}")
            rows = (target[:, None] * m4.EV_SIZE + np.arange(m4.EV_SIZE)).ravel()
            mf_fit[rows] = train_and_score(fit, train_ev, clk_d, n_users, n_items, d,
                                           lambda m: m4.score(m, fit, target))
            log(f"    AUC on day {d}: {roc_auc(mf_fit[rows, 0], fit['label'][rows]):.5f}")
        else:
            log(f"  MF trained on days 0-{d-1} -> scoring the test events")
            mf_test = train_and_score(fit, train_ev, clk_d, n_users, n_items, d,
                                      lambda m: m4.score(m, test, np.arange(len(test["user"]))))
    np.savez(a.out, mf_fit=mf_fit, mf_test=mf_test)


if __name__ == "__main__":
    main()
