#!/usr/bin/env python3
"""
TechArena CTR -- experiment 06: experiment 05 with the MF averaged over several seeds.

Same pipeline as experiment 05 (experiments/03_timesplit/timesplit.py with experiment 05's
feature build incl. the click log); only the MF stage changes (mf_stage.py here).

Usage:
    .venv/bin/python experiments/06_mf_seeds/mf_seeds.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv
    .venv/bin/python experiments/06_mf_seeds/mf_seeds.py --cutoff 7 \
        --train data/train.csv --evaluation data/evaluation.csv --out submissions/06_mf_seeds.csv
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
    t3.MF_TAG = "06"
    t3.MF_DEPS = [HERE.parent / "05_click_aug" / "mf_stage.py", HERE.parent / "04_history_mf" / "mf_stage.py"]
    e2.MF_FEATURES = e5.MF_FEATURES
    t3.main()
