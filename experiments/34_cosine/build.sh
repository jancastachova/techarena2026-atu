#!/usr/bin/env bash
# Experiment 34 submission: the 26_ensemble_bundle recipe (24_ensemble/ensemble.py, LightGBM + XGBoost + MLP, rank
# average) + the sqrt-set view (33) + the cosine view (34_cosine/cos_view.py) + the eight cosine audience views
# (26_audience/audience.py --variants --hops --cosine, 5 vector seeds). Half B vs the bundle's LightGBM feature set:
# +0.00160 [+0.00134, +0.00188], evaluation-weighted +0.00125 [+0.00085, +0.00173].
#     bash experiments/34_cosine/build.sh <model seed> [--mf-core-extra data/cache/26p_mf_c7_5e7ac6880392.npz]
set -euo pipefail
SEED=${1:-0}; shift || true
PY=.venv/bin/python
source experiments/lab/bundle_args.sh
EXTRAS=()
for ((i = 0; i < ${#BUNDLE[@]}; i += 2)); do EXTRAS+=(--extra "${BUNDLE[i+1]}"); done
EXTRAS+=(--extra sqrtview=data/cache/33_sqrtpool_c7.npz --extra cos=data/cache/34_cos_c7.npz)
for k in aud audc audc05 wb hop2 hop2c hop3 hop4; do EXTRAS+=(--extra cos_$k=data/cache/34a_${k}_c7.npz); done
TAG=s$SEED; [ $# -gt 0 ] && TAG=${TAG}_mf45
$PY experiments/24_ensemble/ensemble.py --members lgb_full,xgb_full,mlp_full --seq-caches "$SEQ" "${EXTRAS[@]}" \
    --model-seed "$SEED" "$@" --out submissions/34_cosine_$TAG.csv
$PY src/check_submission.py submissions/34_cosine_$TAG.csv
