#!/usr/bin/env bash
# Experiment 35 submission: experiment 34's recipe with the projected cosine view (34_cosine/cos_view.py --proj 8: the
# vectors projected onto their 8 strong principal directions before the cosine) and the cosine audience views rebuilt
# with UNIT clicker tastes (26_audience/audience.py --cosine --unit-clickers, caches 34b_*). Half B on top of the 34
# set: +0.00110 [+0.00085, +0.00132] (projection) then +0.00034 [+0.00016, +0.00051] (unit clickers); evaluation-
# weighted 0.82569 -> 0.82718.
#     bash experiments/35_projection/build.sh <model seed> [--mf-core-extra data/cache/26p_mf_c7_5e7ac6880392.npz]
set -euo pipefail
SEED=${1:-0}; shift || true
PY=.venv/bin/python
source experiments/lab/bundle_args.sh
EXTRAS=()
for ((i = 0; i < ${#SET35[@]}; i += 2)); do EXTRAS+=(--extra "${SET35[i+1]}"); done
TAG=s$SEED; [ $# -gt 0 ] && TAG=${TAG}_mf45
$PY experiments/24_ensemble/ensemble.py --members lgb_full,xgb_full,mlp_full --seq-caches "$SEQ" "${EXTRAS[@]}" \
    --model-seed "$SEED" "$@" --out submissions/35_projection_$TAG.csv
$PY src/check_submission.py submissions/35_projection_$TAG.csv
