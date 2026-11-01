#!/usr/bin/env bash
# Experiment 41 screen: SET40 with its projected view rebuilt from the cosine-decay vectors (41_cosproj8), paired vs SET40 as is.
# TEST_DAY=6 / 5 to screen, TEST_DAY=4 (unselected holdout) to confirm. Keep only with agreement on all three (~+0.0005 eval-weighted).
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-30_41_screen_d$T.txt
source experiments/lab/bundle_args.sh
NEW=(); for x in "${SET40[@]}"; do [ "$x" = "cosproj8=data/cache/40_cosproj8_c7.npz" ] && x="cosproj8=data/cache/41_cosproj8_c7.npz"; NEW+=("$x"); done
BASEP=data/cache/lab_preds_40_d$T.npz; [ $T = 4 ] && BASEP=data/cache/lab_preds_audit_set40_d4.npz
{
[ -f $BASEP ] || $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" --save-preds $BASEP
$PY experiments/lab/extra_member.py --seq "$SEQ" "${NEW[@]}" --save-preds data/cache/lab_preds_41_d$T.npz
$PY experiments/lab/paired_ci.py "$BASEP:base (= the best member)" "data/cache/lab_preds_41_d$T.npz:base (= the best member)"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
