#!/usr/bin/env bash
# 2026-11-03 fourth box: the LIGHTGBM LAMBDARANK member built straight on the evaluation (SET45 features, random 500-event
# groups, fixed 2200 trees as 37's build, model seed 0), to be rank-averaged into 47s6 at 1/8 at home (48). A gamble: on
# SET35/36 it gave +0.0004-0.0006 on days 6 and 5; inside 47 it is untested (the lab screen did not finish).
set -uo pipefail
cd /workspace/t; PY=.venv/bin/python
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
EXTRAS=(); for x in "${SET45[@]}"; do case "$x" in *=*) EXTRAS+=(--extra "$x");; esac; done
mkdir -p experiments/logs/2026-11-03_box2
NTHREADS=40 RANK_GROUP=500 RANK_TREES=2200 $PY experiments/24_ensemble/ensemble.py --members lgbrank_full --seq-caches "$SEQ" \
    "${EXTRAS[@]}" --model-seed 0 --out submissions/48_rank_s0.csv > experiments/logs/2026-11-03_box2/rank_s0.txt 2>&1
$PY src/check_submission.py submissions/48_rank_s0.csv >> experiments/logs/2026-11-03_box2/rank_s0.txt 2>&1
echo "rank done $(date)"
