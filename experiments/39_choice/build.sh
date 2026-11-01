#!/usr/bin/env bash
# Experiment 39: a NEURAL CHOICE MODEL member (lab/choice_model.py; 24_ensemble/ensemble.py member choice_full): one softmax
# per event over its 10 candidates + "no click", candidate scores from an MLP on the Gaussian-rank features, the no-click
# score from an MLP on the event-constant columns; 3 seeds; epochs by early stopping on day 6, refit on days 2-6. Features:
# the LEAN set (lab/bundle_args.sh SETLEAN, 15 view groups; the full SET38 gives the same choice-model AUC). Lab, day 6 half
# B, 3 seeds: 0.84508 / eval-weighted 0.82833 vs the lean LightGBM 0.84447 / 0.82779; day 5 (1 seed) +0.00037 / +0.00039;
# equal rank blend with LightGBM day 6 0.84508 / 0.82841, with the full-set LightGBM 0.84518 / 0.82855
# (logs 2026-10-30_lab_choice_*). Writes the choice member alone; 39b = equal mean with 38b (15_more_seeds/average.py).
#     bash experiments/39_choice/build.sh <model seed>
set -euo pipefail
SEED=${1:-0}; shift || true
PY=.venv/bin/python
source experiments/lab/bundle_args.sh
EXTRAS=()
for ((i = 0; i < ${#SETLEAN[@]}; i += 2)); do EXTRAS+=(--extra "${SETLEAN[i+1]}"); done
$PY experiments/24_ensemble/ensemble.py --members choice_full --seq-caches "$SEQ" "${EXTRAS[@]}" --model-seed "$SEED" "$@" \
    --out submissions/39_choice_s$SEED.csv
$PY src/check_submission.py submissions/39_choice_s$SEED.csv
$PY experiments/15_more_seeds/average.py submissions/39_choice_s$SEED.csv submissions/38b_avg_38_37c.csv \
    --out submissions/39b_avg_39_38b.csv
$PY src/check_submission.py submissions/39b_avg_39_38b.csv
