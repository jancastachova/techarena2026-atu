#!/usr/bin/env python3
"""
TechArena CTR -- experiment 07: experiment 06 with the MF averaged over 10 seeds (06: 5).

Usage:
    .venv/bin/python experiments/07_mf_seeds10/mf_seeds10.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv
    .venv/bin/python experiments/07_mf_seeds10/mf_seeds10.py --cutoff 7 \
        --train data/train.csv --evaluation data/evaluation.csv --out submissions/07_mf_seeds10.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "05_click_aug"))
import click_aug as e5                # noqa: E402  (experiment 05: build with click log; no torch)

t3, e2 = e5.t3, e5.e2

if __name__ == "__main__":
    t3.build = e5.build
    t3.MF_STAGE = HERE / "mf_stage.py"
    t3.MF_TAG = "07"
    t3.MF_DEPS = [HERE.parent / "06_mf_seeds" / "mf_stage.py", HERE.parent / "05_click_aug" / "mf_stage.py",
                  HERE.parent / "04_history_mf" / "mf_stage.py"]
    e2.MF_FEATURES = e5.MF_FEATURES
    t3.main()
