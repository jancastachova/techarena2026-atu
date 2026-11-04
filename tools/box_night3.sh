#!/usr/bin/env bash
# 2026-11-03 box: the SPECTRAL view (47_spec, 16 raw columns) as LightGBM features: SET45 + spec vs SET45 (night1 lgb), days
# 6/5/4/3, paired.
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
for T in 6 5 4 3; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET45[@]}" --base-extra spec=data/cache/47_spec_c7.npz \
      --save-preds $C/d${T}_lgbSP.npz 2>&1 | grep -E "base \(=|Error|Trace" > $L/lgbSP_d$T.txt ) &
done
wait
for T in 6 5 4 3; do
  { echo "## day $T LightGBM: SET45 (A) vs + spectral (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_lgb.npz:base (= the best member)" "$C/d${T}_lgbSP.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted"
  } >> $L/compare_lgbspec.txt
done
cat $L/compare_lgbspec.txt
