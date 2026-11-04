#!/usr/bin/env bash
# (d) = 45 set + a SECOND audience view from the lr-decayed 41 vectors (soft_aud41 <- 41c_aud, extra group), vs 45 set,
# days 6/5/4 (2026-11-01).
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24
source experiments/lab/bundle_args.sh
D=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; D+=("$x"); done; D+=(--base-extra soft_aud41=data/cache/41c_aud_c7.npz)
L=experiments/logs/2026-11-01_box_aud41x; mkdir -p $L
for T in 6 5 4; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${D[@]}" --save-preds data/cache/lab_preds_aud41x_d$T.npz 2>&1 | grep "base (=" > $L/d$T.txt ) &
done
wait
for T in 6 5 4; do
  echo "### day $T: (d) vs 45 set" >> $L/compare.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_aud40a_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_aud41x_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done
cat $L/compare.txt
