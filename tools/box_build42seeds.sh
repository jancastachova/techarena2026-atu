#!/usr/bin/env bash
# Experiment 42 with more model seeds (2026-11-01, vast.ai box): seeds 1 and 2 of the 4-member 42 ensemble (lgb_full,
# xgb_full, choice_full, lgbshallow_full on SET40), in parallel, 32 threads each, XGBoost on the GPU. Seed 0 is the Mac's
# 40_long_mf_s0 (3 members) + 42_shallow_member_s0, i.e. 42_shallow_s0.csv.
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=32 XGB_DEVICE=cuda
source experiments/lab/bundle_args.sh
EXTRAS=(); for ((i = 0; i < ${#SET40[@]}; i += 2)); do EXTRAS+=(--extra "${SET40[i+1]}"); done
mkdir -p submissions
for S in 1 2; do
  ( $PY experiments/24_ensemble/ensemble.py --members lgb_full,xgb_full,choice_full,lgbshallow_full --seq-caches "$SEQ" "${EXTRAS[@]}" \
      --model-seed $S --out submissions/42_shallow4_s$S.csv > experiments/logs/2026-11-01_42_build_s$S.txt 2>&1
    $PY src/check_submission.py submissions/42_shallow4_s$S.csv >> experiments/logs/2026-11-01_42_build_s$S.txt 2>&1 ) &
done
wait
tail -4 experiments/logs/2026-11-01_42_build_s*.txt
