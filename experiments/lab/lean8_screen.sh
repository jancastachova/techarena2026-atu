#!/usr/bin/env bash
# SET40L (8 groups) vs SET40 (15 groups) on days 6, 5 and 4 -- a simplification is kept if it costs nothing on any day.
set -euo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-10-30_lab_lean8.txt
source experiments/lab/bundle_args.sh
{
for T in 6 5 4; do
  BASE=data/cache/lab_preds_40_d$T.npz; [ $T = 4 ] && BASE=data/cache/lab_preds_audit_set40_d4.npz
  echo "### day $T"
  TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40L[@]}" --save-preds data/cache/lab_preds_40L_d$T.npz | grep "base (="
  $PY experiments/lab/paired_ci.py "$BASE:base (= the best member)" "data/cache/lab_preds_40L_d$T.npz:base (= the best member)" | grep -E "plain|eval-weighted"
done
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
