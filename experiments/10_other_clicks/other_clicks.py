#!/usr/bin/env python3
"""
TechArena CTR -- experiment 10: experiment 09 with the user's OTHER clicks (not only earlier
ones) as context for the click-log examples (see mf_stage.py here). Pipeline otherwise
unchanged (experiment 09's feature build with event timestamps).

Usage:
    .venv/bin/python experiments/10_other_clicks/other_clicks.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv
    .venv/bin/python experiments/10_other_clicks/other_clicks.py --cutoff 7 \
        --train data/train.csv --evaluation data/evaluation.csv --out submissions/10_other_clicks.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "09_full_history"))
import full_history as e9             # noqa: E402  (experiment 09: build with timestamps; no torch)

t3, e2 = e9.t3, e9.e2

if __name__ == "__main__":
    t3.DailyState = e9.TsState
    t3.build = e9.build
    t3.MF_STAGE = HERE / "mf_stage.py"
    t3.MF_TAG = "10"
    t3.MF_DEPS = [HERE.parent / "09_full_history" / "mf_stage.py", HERE.parent / "05_click_aug" / "mf_stage.py",
                  HERE.parent / "04_history_mf" / "mf_stage.py"]
    e2.MF_FEATURES = e9.MF_FEATURES
    t3.main()
