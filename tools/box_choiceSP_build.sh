#!/usr/bin/env bash
# 2026-11-03 box: the 45 build's CHOICE member with the SPECTRAL view added (SET45 + 47_spec_c7.npz), full fit -> evaluation,
# cached as 24_choice_full_sp_s<seed>_test.npy for 43_choice_weight/swap_choice.py. Build settings (3 choice seeds, GPU).
set -uo pipefail
PY=.venv/bin/python; export CHOICE_DEVICE=cuda CHOICE_TAG=_sp NTHREADS=8
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
EXTRAS=(); for x in "${SET45[@]}"; do case "$x" in *=*) EXTRAS+=(--extra "$x");; esac; done
EXTRAS+=(--extra spec=data/cache/47_spec_c7.npz)
mkdir -p experiments/logs/2026-11-03_choiceSP
for S in ${1:-0 1 2}; do
  [ -f data/cache/24_choice_full_sp_s${S}_test.npy ] && continue
  $PY experiments/24_ensemble/ensemble.py --members choice_full --seq-caches "$SEQ" "${EXTRAS[@]}" --model-seed $S \
      --out /tmp/choiceSP_s$S.csv > experiments/logs/2026-11-03_choiceSP/s$S.txt 2>&1
done
grep -h "choice seed\|Error\|Traceback" experiments/logs/2026-11-03_choiceSP/s*.txt
