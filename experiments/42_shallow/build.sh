#!/usr/bin/env bash
# Experiment 42: a DIVERSITY member -- shallow regularised LightGBM (15 leaves, min_data_in_leaf 2000, feature_fraction 0.3;
# member lgbshallow_full) on 40's feature set SET40. Lab (lab/diversity_screen.sh W2), 50/50 rank average with the SET40
# LightGBM vs SET40 alone, eval- / gap-weighted: day 6 +0.00013 / +0.00010, day 5 +0.00019 / +0.00016, day 4 (unselected)
# +0.00027 / +0.00031; alone it beats SET40 on day 4 (+0.00030 / +0.00035). Added as a 4th equal member of 40
# (40 x3 + shallow x1 in average.py = 4 equal rank members), then 42b = equal mean with 39b, as 40b was built.
#     bash experiments/42_shallow/build.sh <model seed>
set -euo pipefail
SEED=${1:-0}; shift || true
PY=.venv/bin/python
source experiments/lab/bundle_args.sh
EXTRAS=()
for ((i = 0; i < ${#SET40[@]}; i += 2)); do EXTRAS+=(--extra "${SET40[i+1]}"); done
$PY experiments/24_ensemble/ensemble.py --members lgbshallow_full --seq-caches "$SEQ" "${EXTRAS[@]}" \
    --model-seed "$SEED" "$@" --out submissions/42_shallow_member_s$SEED.csv
$PY src/check_submission.py submissions/42_shallow_member_s$SEED.csv
$PY experiments/15_more_seeds/average.py submissions/40_long_mf_s0.csv submissions/40_long_mf_s0.csv submissions/40_long_mf_s0.csv \
    submissions/42_shallow_member_s$SEED.csv --out submissions/42_shallow_s$SEED.csv
$PY src/check_submission.py submissions/42_shallow_s$SEED.csv
$PY experiments/15_more_seeds/average.py submissions/42_shallow_s$SEED.csv submissions/39b_avg_39_38b.csv \
    --out submissions/42b_avg_42_39b.csv
$PY src/check_submission.py submissions/42b_avg_42_39b.csv
