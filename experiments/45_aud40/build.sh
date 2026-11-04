#!/usr/bin/env bash
# Experiment 45 (2026-11-01): SET40 with the soft-projected AUDIENCE view rebuilt from the 9000-step 40 vectors (soft_aud <-
# 40c_aud_c7.npz: 26_audience/audience.py --cosine --unit-clickers --soft-proj 20, 5 seeds of 40_vectors; day 6 alone 0.82855
# vs 0.82726). A representation change. Lab (tools/box_aud40.sh, paired vs box SET40, plain / eval- / gap-weighted): day 6
# +0.00005 / +0.00004 / +0.00021, day 5 +0.00022 / +0.00023 / +0.00028, day 4 +0.00029 / +0.00033 / +0.00023.
# Members as 42 (lgb, xgb, choice, shallow), choice on the GPU, cached as 24_choice_full_45_s<seed>_test.npy so that
# 43_choice_weight/mix_choice.py can lift the choice member to 0.40.
#     bash experiments/45_aud40/build.sh "0 1 2"
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=${NTHREADS:-32} XGB_DEVICE=cuda CHOICE_DEVICE=cuda CHOICE_TAG=_45
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
EXTRAS=(); for x in "${SET45[@]}"; do case "$x" in *=*) EXTRAS+=(--extra "$x");; esac; done
mkdir -p submissions
for S in ${1:-0 1 2}; do
  ( $PY experiments/24_ensemble/ensemble.py --members lgb_full,xgb_full,choice_full,lgbshallow_full --seq-caches "$SEQ" "${EXTRAS[@]}" \
      --model-seed $S --out submissions/45_aud40_s$S.csv > experiments/logs/2026-11-01_45_build_s$S.txt 2>&1
    $PY src/check_submission.py submissions/45_aud40_s$S.csv >> experiments/logs/2026-11-01_45_build_s$S.txt 2>&1 ) &
done
wait
grep -h "OK\|Error\|Traceback\|trees\|choice seed" experiments/logs/2026-11-01_45_build_s*.txt
