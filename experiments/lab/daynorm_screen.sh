#!/usr/bin/env bash
# Day-normalised MF scores (lab/extra_member.py DAYNORM=z) on SET38, paired vs SET38 as is (screen 3's bundle preds).
# TEST_DAY=5 runs the fresh day-5 holdout.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-30_lab_daynorm_d$T.txt
source experiments/lab/bundle_args.sh
{
DAYNORM=z $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET38[@]}" --save-preds data/cache/lab_preds_daynorm_d$T.npz
echo "### paired: DAYNORM=z vs SET38"
$PY experiments/lab/paired_ci.py "data/cache/lab_preds_38_act_d$T.npz:base (= the best member)" "data/cache/lab_preds_daynorm_d$T.npz:base (= the best member)"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
