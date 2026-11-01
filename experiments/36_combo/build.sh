#!/usr/bin/env bash
# Experiment 36 submission: experiment 35 plus four small views that together pass a paired interval on the evaluation-
# weighted half B (lab/extra_member.py, log 2026-10-28_lab_small_combo.txt): temperature-0.6 projected cosine, the cosine
# of the 24 directions the 8-dim projection drops (lab/replica_gap.py: real-only signal), soft projection K=20 and the
# soft-projected audience; and the LightGBM member trained by lambdarank over groups of 1,000 events (lab/global_rank2.py:
# +0.00025 [+0.00012, +0.00038] half B, +0.00043 [+0.00018, +0.00070] eval-weighted on SET35). Views alone: half B +0.00007 [-0.00007, +0.00023], eval-weighted +0.00035 [+0.00005, +0.00069].
#     bash experiments/36_combo/build.sh <model seed>
set -euo pipefail
SEED=${1:-0}; shift || true
PY=.venv/bin/python
source experiments/lab/bundle_args.sh
EXTRAS=()
for ((i = 0; i < ${#SET36[@]}; i += 2)); do EXTRAS+=(--extra "${SET36[i+1]}"); done
TAG=s$SEED; [ $# -gt 0 ] && TAG=${TAG}_mf45
$PY experiments/24_ensemble/ensemble.py --members lgbrank_full,xgb_full,mlp_full --seq-caches "$SEQ" "${EXTRAS[@]}" \
    --model-seed "$SEED" "$@" --out submissions/36_combo_$TAG.csv
$PY src/check_submission.py submissions/36_combo_$TAG.csv
