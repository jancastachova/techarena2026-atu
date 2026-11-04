#!/usr/bin/env bash
# Final-build prep (2026-11-01, box GPU): the CHOICE member alone on the full 36-group set (SET38 with the 40 view), model seeds
# 0-2, 5 choice seeds each on the GPU (CHOICE_DEVICE=cuda), cached as 24_choice_full_d38_s<seed>_test.npy for
# 43_choice_weight/mix_choice.py. Lab: in the ensemble the D38 choice beats the SET40 one by +0.00002..+0.00007 on days 6/5/4.
set -uo pipefail
PY=.venv/bin/python; export CHOICE_DEVICE=cuda CHOICE_SEEDS=5 CHOICE_TAG=_d38
source experiments/lab/bundle_args.sh
D38=(); for x in "${SET38[@]}"; do [ "$x" = "cosproj8=data/cache/38_cosproj8_c7.npz" ] && x="cosproj8=data/cache/40_cosproj8_c7.npz"; D38+=("$x"); done
EXTRAS=(); for x in "${D38[@]}"; do case "$x" in *=*) EXTRAS+=(--extra "$x");; esac; done
for S in ${SEEDS:-0 1 2}; do
  $PY experiments/24_ensemble/ensemble.py --members choice_full --seq-caches "$SEQ" "${EXTRAS[@]}" --model-seed $S \
      --out submissions/44_choice_d38_s$S.csv > experiments/logs/2026-11-01_44_choice_d38_s$S.txt 2>&1
done
grep -h "choice seed\|done\|Error" experiments/logs/2026-11-01_44_choice_d38_s*.txt
