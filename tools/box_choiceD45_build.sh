#!/usr/bin/env bash
# 2026-11-03 box: the CHOICE member of the 45 build trained on D45 (SET38 with 40's projected and audience views, 337
# features) instead of SET45, full fit (days 2-6) -> evaluation, cached as 24_choice_full_d45_s<seed>_test.npy for
# 43_choice_weight/swap_choice.py. Same choice settings and seeds as the 45 build (3 choice seeds per model seed, GPU).
set -uo pipefail
PY=.venv/bin/python; export CHOICE_DEVICE=cuda CHOICE_TAG=_d45 NTHREADS=8
source experiments/lab/bundle_args.sh
D45=(); for x in "${SET38[@]}"; do
  [ "$x" = "cosproj8=data/cache/38_cosproj8_c7.npz" ] && x="cosproj8=data/cache/40_cosproj8_c7.npz"
  [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; D45+=("$x"); done
EXTRAS=(); for x in "${D45[@]}"; do case "$x" in *=*) EXTRAS+=(--extra "$x");; esac; done
mkdir -p experiments/logs/2026-11-03_choiceD45
for S in ${1:-0 1 2}; do
  [ -f data/cache/24_choice_full_d45_s${S}_test.npy ] && continue
  $PY experiments/24_ensemble/ensemble.py --members choice_full --seq-caches "$SEQ" "${EXTRAS[@]}" --model-seed $S \
      --out /tmp/choiceD45_s$S.csv > experiments/logs/2026-11-03_choiceD45/s$S.txt 2>&1
done
grep -h "choice seed\|Error\|Traceback" experiments/logs/2026-11-03_choiceD45/s*.txt
