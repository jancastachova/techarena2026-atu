#!/usr/bin/env bash
# Experiment 37 submission: experiment 36's feature set (SET36) with FOUR members -- binary LightGBM, lambdarank LightGBM
# over random groups of 500 events, XGBoost, MLP -- rank-averaged. Safer than replacing the binary LightGBM (lambdarank
# weights the top of each group; global AUC weighs all pairs). lab/global_rank2.py on SET36: G1000 alone 0.84458 / 0.82777,
# rank average with binary 0.84456 / 0.82781 (half B / eval-weighted); G500 0.84457 / 0.82776 at half the gradient cost.
# Day-5 fresh holdout (fit days 2-4) confirms: SET35 binary -> SET36 binary + G500 rank average +0.00050 [+0.00021, +0.00074]
# eval-weighted, first events +0.00047. The ranking member uses a fixed 2200 trees (RANK_TREES) to skip early stopping.
#     bash experiments/37_rank4/build.sh <model seed>
set -euo pipefail
SEED=${1:-0}; shift || true
PY=.venv/bin/python
source experiments/lab/bundle_args.sh
EXTRAS=()
for ((i = 0; i < ${#SET36[@]}; i += 2)); do EXTRAS+=(--extra "${SET36[i+1]}"); done
RANK_GROUP=500 RANK_TREES=${RANK_TREES:-2200} $PY experiments/24_ensemble/ensemble.py --members lgb_full,lgbrank_full,xgb_full,mlp_full --seq-caches "$SEQ" \
    "${EXTRAS[@]}" --model-seed "$SEED" "$@" --out submissions/37_rank4_s$SEED.csv
$PY src/check_submission.py submissions/37_rank4_s$SEED.csv
