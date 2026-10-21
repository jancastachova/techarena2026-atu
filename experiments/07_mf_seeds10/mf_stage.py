#!/usr/bin/env python3
"""
Matrix factorisation stage of experiment 07: experiment 06's MF stage with 10 seeds instead
of 5. Everything else is unchanged.

Chosen inside the fit period only (MF trained on days 0-3 + clicks before day 4, AUC on
fit day 4; cutoff-5 base features):
    seeds      1      3      5      8      10
    day-4 AUC  0.736  0.758  0.764  0.768  0.768

Usage (normally called by mf_seeds10.py):
    .venv/bin/python experiments/07_mf_seeds10/mf_stage.py --base <base.npz> --out <mf.npz> --cutoff 5
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# experiment 06's MF stage, loaded by path (same file name as this one)
_spec = importlib.util.spec_from_file_location(
    "mf_stage_06", ROOT / "experiments" / "06_mf_seeds" / "mf_stage.py")
m6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m6)

N_SEEDS = 10

if __name__ == "__main__":
    m6.N_SEEDS = N_SEEDS
    m6.main()
