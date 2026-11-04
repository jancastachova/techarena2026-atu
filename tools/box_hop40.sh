#!/usr/bin/env bash
# (g) = 45 set with ONLY cos_hop4 rebuilt from the 40 vectors (40b_hop4; (b) on 2026-11-01 swapped hop4 and hop2c together),
# vs 45 set, days 6/5/4 (2026-11-02).
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24
source experiments/lab/bundle_args.sh
D=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; D+=("$x"); done; G=(); for x in "${D[@]}"; do [ "$x" = "cos_hop4=data/cache/34b_hop4_c7.npz" ] && x="cos_hop4=data/cache/40b_hop4_c7.npz"; G+=("$x"); done; D=("${G[@]}")
L=experiments/logs/2026-11-02_box_hop40; mkdir -p $L
for T in 6 5 4; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${D[@]}" --save-preds data/cache/lab_preds_hop40_d$T.npz 2>&1 | grep "base (=" > $L/d$T.txt ) &
done
wait
for T in 6 5 4; do
  echo "### day $T: (g) vs 45 set" >> $L/compare.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_aud40a_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_hop40_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done
cat $L/compare.txt
