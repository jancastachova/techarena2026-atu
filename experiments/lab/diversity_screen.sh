#!/usr/bin/env bash
# ENSEMBLE-DIVERSITY screen (2026-10-31): candidate members trained on the lab split, each rank-averaged 50/50 with the SET40
# lab model and paired vs SET40 alone (plain / eval-weighted / gap-weighted). A member is worth a build only if the AVERAGE
# beats SET40 on days 6 and 5 and then on day 4. TEST_DAY=6/5/4; MEMBERS picks candidates.
#   W1   SET40, extra-trees LightGBM (random thresholds, feature_fraction 0.5, 127 leaves)
#   W2   SET40, shallow regularised LightGBM (15 leaves, min_data_in_leaf 2000, feature_fraction 0.3)
#   D38  the full 36-group set (SET38) with the 40 projected view
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-31_lab_diversity_d$T.txt
source experiments/lab/bundle_args.sh
D38=(); for x in "${SET38[@]}"; do [ "$x" = "cosproj8=data/cache/38_cosproj8_c7.npz" ] && x="cosproj8=data/cache/40_cosproj8_c7.npz"; D38+=("$x"); done
BASEP=data/cache/lab_preds_40_d$T.npz; [ $T = 4 ] && BASEP=data/cache/lab_preds_audit_set40_d4.npz
{
for M in ${MEMBERS:-W1 W2 D38}; do
  P=data/cache/lab_preds_div_${M}_d$T.npz
  echo "### $M"
  case $M in
    W1) LGBP='{"extra_trees": true, "feature_fraction": 0.5, "num_leaves": 127}' $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" --save-preds $P | grep "base (=";;
    W2) LGBP='{"num_leaves": 15, "min_data_in_leaf": 2000, "feature_fraction": 0.3}' $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" --save-preds $P | grep "base (=";;
    D38) $PY experiments/lab/extra_member.py --seq "$SEQ" "${D38[@]}" --save-preds $P | grep "base (=";;
  esac
  $PY experiments/lab/avg_preds.py data/cache/lab_preds_div_${M}avg_d$T.npz $BASEP $P
  echo "member alone vs SET40:"; $PY experiments/lab/paired_ci.py "$BASEP:base (= the best member)" "$P:base (= the best member)" | grep -E "plain|weighted"
  echo "50/50 average vs SET40:"; $PY experiments/lab/paired_ci.py "$BASEP:base (= the best member)" "data/cache/lab_preds_div_${M}avg_d$T.npz:base (= the best member)" | grep -E "plain|weighted"
done
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
