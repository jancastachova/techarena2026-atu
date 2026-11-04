#!/usr/bin/env bash
# Unfinished representation lead (2026-11-01): audience / hop views rebuilt from the 40 vectors (40_long_mf/screen2.sh; day 6
# only: (a) +0.00014 / +0.00005, (b) +0.00012 / +0.00005 plain / eval-weighted; day 5 never finished). Now days 6/5/4 on the
# box, paired vs box SET40.  (a) soft_aud <- 40c_aud   (b) (a) + cos_hop4 / cos_hop2c <- 40b_hop4 / 40b_hop2c
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=${NTHREADS:-12}
source experiments/lab/bundle_args.sh
A=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; A+=("$x"); done
B=(); for x in "${A[@]}"; do
  [ "$x" = "cos_hop4=data/cache/34b_hop4_c7.npz" ] && x="cos_hop4=data/cache/40b_hop4_c7.npz"
  [ "$x" = "cos_hop2c=data/cache/34b_hop2c_c7.npz" ] && x="cos_hop2c=data/cache/40b_hop2c_c7.npz"; B+=("$x"); done
L=experiments/logs/2026-11-01_box_aud40; mkdir -p $L
for T in 6 5 4; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${A[@]}" --save-preds data/cache/lab_preds_aud40a_d$T.npz 2>&1 | grep "base (=" > $L/a_d$T.txt ) &
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${B[@]}" --save-preds data/cache/lab_preds_aud40b_d$T.npz 2>&1 | grep "base (=" > $L/b_d$T.txt ) &
done
wait
for T in 6 5 4; do for V in a b; do
  echo "### day $T: ($V) vs box SET40" >> $L/compare.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_boxbase_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_aud40${V}_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done; done
cat $L/compare.txt
