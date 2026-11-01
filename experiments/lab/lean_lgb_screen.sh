#!/usr/bin/env bash
# LightGBM capacity on the LEAN set (SETLEAN): default lab settings (63 leaves, lr 0.02) vs bigger / more regularised trees,
# each paired vs the lean default. TEST_DAY=5 for the fresh holdout; VARIANTS overrides the list (name|json;...).
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-30_lab_lean_lgb_d$T.txt
source experiments/lab/bundle_args.sh
V=${VARIANTS:-'l127|{"num_leaves":127};l255|{"num_leaves":255,"min_data_in_leaf":100};ff05|{"feature_fraction":0.5};l127l2|{"num_leaves":127,"lambda_l2":10,"min_data_in_leaf":400};lr01|{"learning_rate":0.01}'}
{
echo "### lean default"
$PY experiments/lab/extra_member.py --seq "$SEQ" "${SETLEAN[@]}" --save-preds data/cache/lab_preds_lean0_d$T.npz
IFS=';' read -ra VS <<< "$V"
for v in "${VS[@]}"; do n=${v%%|*}; j=${v#*|}
  echo "### $n $j"
  LGBP="$j" $PY experiments/lab/extra_member.py --seq "$SEQ" "${SETLEAN[@]}" --save-preds data/cache/lab_preds_lean_${n}_d$T.npz
  $PY experiments/lab/paired_ci.py "data/cache/lab_preds_lean0_d$T.npz:base (= the best member)" "data/cache/lab_preds_lean_${n}_d$T.npz:base (= the best member)"
done
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
