#!/usr/bin/env bash
# GAP WEIGHTING screen (2026-10-31): SET40 with LightGBM training events weighted toward evaluation's hours-since-last-click mix
# (GAPW=p in lab/extra_member.py; early stopping on gap-weighted half A), paired vs SET40 unweighted. TEST_DAY=6 / 5.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-31_lab_gapw_d$T.txt
source experiments/lab/bundle_args.sh
BASEP=data/cache/lab_preds_40_d$T.npz
{
for P in ${POWERS:-1 0.5}; do
  echo "### GAPW=$P"
  GAPW=$P $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" --save-preds data/cache/lab_preds_gapw${P}_d$T.npz | grep -E "GAPW|base \(="
  $PY experiments/lab/paired_ci.py "$BASEP:base (= the best member)" "data/cache/lab_preds_gapw${P}_d$T.npz:base (= the best member)" | grep -E "plain|weighted"
done
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
