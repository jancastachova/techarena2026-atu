#!/usr/bin/env python3
"""
TechArena CTR -- experiment 04: experiment 03 with the history-only MF (mf_stage.py here).

Everything else is experiment 03 unchanged (experiments/03_timesplit/timesplit.py): the
same time split, base features, LightGBM procedure and segment report, so a change in the
score comes from the MF alone.

Usage:
    # validation (cutoff day 5, no submission written)
    .venv/bin/python experiments/04_history_mf/history_mf.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv
    # submission
    .venv/bin/python experiments/04_history_mf/history_mf.py --cutoff 7 \
        --train data/train.csv --evaluation data/evaluation.csv --out submissions/04_history_mf.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "03_timesplit"))
import timesplit as t3                # noqa: E402
import mf_lgbm as e2                  # noqa: E402  (already on sys.path through timesplit)

# Column order written by mf_stage.py (not imported here: it loads torch, which must not
# share a process with lightgbm). A mismatch in count makes LightGBM fail loudly.
MF_FEATURES = ["mf_logit", "mf_item_bias", "mf_user_bias", "mf_hist_dot"]

if __name__ == "__main__":
    t3.MF_STAGE = HERE / "mf_stage.py"
    t3.MF_TAG = "04"
    e2.MF_FEATURES = MF_FEATURES
    t3.main()
