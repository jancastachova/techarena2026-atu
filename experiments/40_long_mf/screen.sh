#!/usr/bin/env bash
# Experiment 40 screen: the lean set (SETLEAN) with its projected view rebuilt from the 9000-step vectors, paired vs the lean
# set as is (lab/lean_lgb_screen.sh's lean default). TEST_DAY=5 for the fresh holdout.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-30_40_screen_d$T.txt
source experiments/lab/bundle_args.sh
NEW=(); for x in "${SETLEAN[@]}"; do [ "$x" = "cosproj8=data/cache/38_cosproj8_c7.npz" ] && x="cosproj8=data/cache/40_cosproj8_c7.npz"; NEW+=("$x"); done
BASEP=data/cache/lab_preds_lean0_d$T.npz
{
[ -f $BASEP ] || $PY experiments/lab/extra_member.py --seq "$SEQ" "${SETLEAN[@]}" --save-preds $BASEP
$PY experiments/lab/extra_member.py --seq "$SEQ" "${NEW[@]}" --save-preds data/cache/lab_preds_40_d$T.npz
$PY experiments/lab/paired_ci.py "$BASEP:base (= the best member)" "data/cache/lab_preds_40_d$T.npz:base (= the best member)"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
