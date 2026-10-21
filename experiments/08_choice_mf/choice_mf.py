#!/usr/bin/env python3
"""
TechArena CTR -- experiment 08: experiment 07 (10-seed MF with click log) with a choice-model
impression loss in the MF (see mf_stage.py here). Pipeline otherwise unchanged.

Usage:
    .venv/bin/python experiments/08_choice_mf/choice_mf.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv
    .venv/bin/python experiments/08_choice_mf/choice_mf.py --cutoff 7 \
        --train data/train.csv --evaluation data/evaluation.csv --out submissions/08_choice_mf.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "05_click_aug"))
import click_aug as e5                # noqa: E402  (experiment 05: build with click log; no torch)

t3, e2 = e5.t3, e5.e2

# Column order written by mf_stage.py; the first column is now the choice log-probability.
MF_FEATURES = ["mf_choice_logp", "mf_item_bias", "mf_user_bias", "mf_hist_dot"]

if __name__ == "__main__":
    t3.build = e5.build
    t3.MF_STAGE = HERE / "mf_stage.py"
    t3.MF_TAG = "08"
    t3.MF_DEPS = [HERE.parent / "05_click_aug" / "mf_stage.py", HERE.parent / "04_history_mf" / "mf_stage.py"]
    e2.MF_FEATURES = MF_FEATURES
    t3.main()
