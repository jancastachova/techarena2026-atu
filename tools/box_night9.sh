#!/usr/bin/env bash
# 2026-11-03 box: LEARNED SPECTRAL SCORE (lab/spectral_score.py -> 47_specw, time-forward weights) as one LightGBM group:
# SET45 + specw vs SET45 (night1 lgb), days 6/5/4/3, paired; and in the 47 composition (lgb swapped, W2/xgb unchanged).
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
$PY experiments/lab/spectral_score.py > experiments/logs/2026-11-03_lab_spectral_score.txt 2>&1
for T in 6 5 4 3; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET45[@]}" --base-extra specw=data/cache/47_specw_c7.npz \
      --save-preds $C/d${T}_lgbW.npz 2>&1 | grep -E "base \(=|Error|Trace" > $L/lgbW_d$T.txt ) &
done
wait
for T in 6 5 4 3; do
  { echo "## day $T LightGBM: SET45 (A) vs + specw (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_lgb.npz:base (= the best member)" "$C/d${T}_lgbW.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  } > $L/lgbW_cmp_d$T.txt &
done
wait
cat $L/lgbW_cmp_d*.txt > $L/compare_lgbW.txt; cat $L/compare_lgbW.txt
