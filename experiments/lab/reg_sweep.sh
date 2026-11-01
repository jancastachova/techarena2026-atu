#!/usr/bin/env bash
# REGULARISATION SWEEP (2026-10-31): the shallow member W2 (15 leaves, min_data_in_leaf 2000, feature_fraction 0.3) beat the
# SET40 LightGBM ALONE on the unselected day 4 (+0.00030 eval-weighted): the main model may overfit. Single models on SET40,
# paired vs SET40 (63 leaves). TEST_DAY=6/5 to screen, 4 to confirm the winner. VARIANTS picks.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-31_lab_regsweep_d$T.txt
source experiments/lab/bundle_args.sh
BASEP=data/cache/lab_preds_40_d$T.npz; [ $T = 4 ] && BASEP=data/cache/lab_preds_audit_set40_d4.npz
declare_p() { case $1 in
  R1) echo '{"num_leaves": 7, "min_data_in_leaf": 2000, "feature_fraction": 0.3}';;
  R2) echo '{"num_leaves": 15, "min_data_in_leaf": 5000, "feature_fraction": 0.3}';;
  R3) echo '{"num_leaves": 31, "min_data_in_leaf": 2000, "feature_fraction": 0.3}';;
  R4) echo '{"num_leaves": 15, "min_data_in_leaf": 2000, "feature_fraction": 0.15}';;
  R5) echo '{"num_leaves": 15, "min_data_in_leaf": 2000, "feature_fraction": 0.3, "lambda_l2": 50}';;
esac; }
{
for V in ${VARIANTS:-R1 R2 R3 R4}; do
  P=data/cache/lab_preds_reg_${V}_d$T.npz
  echo "### $V $(declare_p $V)"
  LGBP="$(declare_p $V)" $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" --save-preds $P | grep "base (="
  $PY experiments/lab/paired_ci.py "$BASEP:base (= the best member)" "$P:base (= the best member)" | grep -E "plain|weighted"
done
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
