#!/usr/bin/env bash
# (c) = 45 set + cos_audc05 <- 40c_audc (alone day 6 0.82735 vs 34b_audc05 0.82679), vs 45 set (40c, lab_preds_aud40a_d*),
# days 6/5/4 (2026-11-01).
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24
source experiments/lab/bundle_args.sh
D=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; [ "$x" = "cos_audc05=data/cache/34b_audc05_c7.npz" ] && x="cos_audc05=data/cache/40c_audc_c7.npz"; D+=("$x"); done
L=experiments/logs/2026-11-01_box_aud40c2; mkdir -p $L
for T in 6 5 4; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${D[@]}" --save-preds data/cache/lab_preds_aud40c2_d$T.npz 2>&1 | grep "base (=" > $L/d$T.txt ) &
done
wait
for T in 6 5 4; do
  echo "### day $T: (c) vs 45 set" >> $L/compare.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_aud40a_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_aud40c2_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done
cat $L/compare.txt
