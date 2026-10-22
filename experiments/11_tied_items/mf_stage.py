#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 11 (run by tied_items.py as a separate process,
because torch and lightgbm crash when loaded into the same process on macOS).

Experiment 10's MF with ONE vector per item instead of two. Until now each item had a vector
for when it appears in a user's history (y) and another for when it is a candidate (q), each
learned from part of the item's few clicks. Items with few clicks were the weak spot (fit day 4,
1 seed: AUC 0.68 for items with 1-4 clicks, 0.73 for 5-9, 0.81 for 40+). Sharing one vector
means every click trains it in both roles.

Chosen inside the fit period only (MF trained on days 0-3 + clicks before day 4, AUC on
fit day 4):
                                   1 seed   3 seeds
    10 (separate y and q)          0.779    0.791
    11 (one vector per item)       0.784    0.795
Also tried on top of 10 and not kept: 49 instead of 9 click-log negatives (0.774, 1 seed);
other clicks instead of earlier clicks as the training user vector for impressions (0.779).

Usage (normally called by tied_items.py):
    .venv/bin/python experiments/11_tied_items/mf_stage.py --base <base.npz> --out <mf.npz> --cutoff 5
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# experiment 10's MF stage, loaded by path (same file name as this one)
_spec = importlib.util.spec_from_file_location(
    "mf_stage_10", ROOT / "experiments" / "10_other_clicks" / "mf_stage.py")
m10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m10)
m4 = m10.m4


class TiedMF(m4.HistoryMF):
    """Experiment 04's model with the candidate vectors tied to the history vectors."""

    def __init__(self, n_users, n_items, dim):
        super().__init__(n_users, n_items, dim)
        self.qi = self.yh                     # yh has padding_idx=0; candidates are never 0


def use_tied_model():
    m4.HistoryMF = TiedMF                     # experiment 05's train() builds m4.HistoryMF


if __name__ == "__main__":
    use_tied_model()
    m10.main()
