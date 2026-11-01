#!/usr/bin/env bash
# Experiment 40 submission: the LEAN set (lab/bundle_args.sh SET40 = SETLEAN, 15 view groups) with the projected cosine view
# rebuilt from an MF trained twice as long (9000 steps, click-log weight 0.3; 40_long_mf/vectors.sh). Lab, lean LightGBM paired
# vs SETLEAN, plain / eval-weighted: day 6 +0.00057 [+0.00041, +0.00076] / +0.00060 [+0.00032, +0.00090]; day-5 fresh holdout
# +0.00053 / +0.00061 (logs 2026-10-30_40_screen_d6/d5.txt). Members: binary LightGBM, XGBoost, the neural choice model
# (39); rank-averaged. No lambdarank member (100 min for ~nothing on public). 40b = equal mean with 39b.
#     bash experiments/40_long_mf/build.sh <model seed>
set -euo pipefail
SEED=${1:-0}; shift || true
PY=.venv/bin/python
source experiments/lab/bundle_args.sh
EXTRAS=()
for ((i = 0; i < ${#SET40[@]}; i += 2)); do EXTRAS+=(--extra "${SET40[i+1]}"); done
$PY experiments/24_ensemble/ensemble.py --members lgb_full,xgb_full,choice_full --seq-caches "$SEQ" "${EXTRAS[@]}" \
    --model-seed "$SEED" "$@" --out submissions/40_long_mf_s$SEED.csv
$PY src/check_submission.py submissions/40_long_mf_s$SEED.csv
$PY experiments/15_more_seeds/average.py submissions/40_long_mf_s$SEED.csv submissions/39b_avg_39_38b.csv \
    --out submissions/40b_avg_40_39b.csv
$PY src/check_submission.py submissions/40b_avg_40_39b.csv
