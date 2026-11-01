#!/usr/bin/env bash
# SET38 trained only on the later fit days (lab/extra_member.py MINDAY=k), paired vs SET38 on fit days 2+. Day 6: k = 3, 4;
# day-5 holdout: k = 3.
set -euo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-10-30_lab_minday.txt
source experiments/lab/bundle_args.sh
{
for spec in 6:3 6:4 5:3; do T=${spec%:*}; K=${spec#*:}
  echo "### TEST_DAY $T, MINDAY $K"
  TEST_DAY=$T MINDAY=$K $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET38[@]}" --save-preds data/cache/lab_preds_minday${K}_d$T.npz
  $PY experiments/lab/paired_ci.py "data/cache/lab_preds_38_act_d$T.npz:base (= the best member)" "data/cache/lab_preds_minday${K}_d$T.npz:base (= the best member)"
done
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
