#!/usr/bin/env bash
# Leaner SET38: rank the 36 view groups by split gain in the full model, refit on the top-k (lab/extra_member.py LEAN=...).
# TEST_DAY=5 runs the fresh day-5 holdout (the ranking is recomputed there).
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-30_lab_lean_d$T.txt
source experiments/lab/bundle_args.sh
LEAN=${LEAN:-0,3,6,10,15,20} $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET38[@]}" --save-preds data/cache/lab_preds_lean_d$T.npz \
    2>&1 | grep -v "^\[LightGBM\]" | tee $L
