#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 03 (run by timesplit.py as a separate process,
because torch and lightgbm crash when loaded into the same process on macOS).

Uses the MF model of experiment 02 (experiments/02_mf_lgbm/mf_model.py). For a cutoff day C:
  - fit rows of day d (2 <= d < C) are scored by an MF trained on days < d,
  - test rows (first event per user at/after C, or evaluation.csv when C = 7) are scored
    by an MF trained on all fit days (< C).

Usage (normally called by timesplit.py):
    .venv/bin/python experiments/03_timesplit/mf_stage.py --base <base.npz> --out <mf.npz> --cutoff 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "02_mf_lgbm"))
sys.path.insert(0, str(ROOT / "src"))
import mf_model as m2                 # noqa: E402
from metrics import roc_auc           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=int, required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)

    fit, test, n_users, n_items = m2.load_base(a.base)
    row_day = np.repeat(fit["day"], m2.EV_SIZE)
    mf_fit = np.full((len(row_day), len(m2.MF_FEATURES)), np.nan, dtype=np.float32)
    for d in range(m2.FIRST_MODEL_DAY, a.cutoff + 1):
        rows = np.flatnonzero(row_day < d)
        if d < a.cutoff:
            target = np.flatnonzero(row_day == d)
            y = fit["label"][target]
            m2.log(f"  MF trained on days 0-{d-1} ({len(rows):,} rows) -> scoring fit day {d}")
            model = m2.train(fit, rows, n_users, n_items)
            mf_fit[target] = m2.score(model, fit, target)
            m2.log(f"    AUC on day {d}: {roc_auc(mf_fit[target, 0], y):.5f}")
        else:
            m2.log(f"  MF trained on days 0-{d-1} ({len(rows):,} rows) -> scoring the test events")
            model = m2.train(fit, rows, n_users, n_items)
            mf_test = m2.score(model, test, np.arange(len(test["item"])))
    np.savez(a.out, mf_fit=mf_fit, mf_test=mf_test)


if __name__ == "__main__":
    main()
