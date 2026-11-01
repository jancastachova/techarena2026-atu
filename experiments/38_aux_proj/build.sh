#!/usr/bin/env bash
# Experiment 38 submission: experiment 37's four members (binary LightGBM, lambdarank LightGBM over random 500-event groups,
# XGBoost, MLP; rank-averaged) on SET38 (lab/bundle_args.sh) = SET36 with the projected view rebuilt from an MF tuned for it
# (click-log weight 0.3, 4500 steps; 38_aux_proj/vectors.sh + 34_cosine/cos_view.py --proj 8) + reopened user-activity and
# scoring-day features (lab/activity_view.py). LightGBM paired vs SET36, plain / eval-weighted: day 6 +0.00037 / +0.00041,
# day-5 fresh holdout +0.00045 / +0.00050 (logs 2026-10-29_38_screen*_d6/d5.txt).
#     bash experiments/38_aux_proj/build.sh <model seed>
set -euo pipefail
SEED=${1:-0}; shift || true
PY=.venv/bin/python
source experiments/lab/bundle_args.sh
EXTRAS=()
for ((i = 0; i < ${#SET38[@]}; i += 2)); do EXTRAS+=(--extra "${SET38[i+1]}"); done
RANK_GROUP=500 RANK_TREES=${RANK_TREES:-2200} $PY experiments/24_ensemble/ensemble.py --members lgb_full,lgbrank_full,xgb_full,mlp_full --seq-caches "$SEQ" \
    "${EXTRAS[@]}" --model-seed "$SEED" "$@" --out submissions/38_aux_act_s$SEED.csv
$PY src/check_submission.py submissions/38_aux_act_s$SEED.csv
