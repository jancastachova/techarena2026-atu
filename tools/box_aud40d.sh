#!/usr/bin/env bash
# 40d (soft-projected audience from all 10 40-vector seeds; day 6 alone 0.82873) vs 40c (5 seeds; 0.82855, in experiment 45),
# SET40 with soft_aud swapped, days 6/5/4, paired vs the 40c run (lab_preds_aud40a_d*).
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24
source experiments/lab/bundle_args.sh
D=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40d_aud_c7.npz"; D+=("$x"); done
L=experiments/logs/2026-11-01_box_aud40d; mkdir -p $L
for T in 6 5 4; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${D[@]}" --save-preds data/cache/lab_preds_aud40d_d$T.npz 2>&1 | grep "base (=" > $L/d$T.txt ) &
done
wait
for T in 6 5 4; do
  echo "### day $T: 40d vs 40c" >> $L/compare.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_aud40a_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_aud40d_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done
cat $L/compare.txt
