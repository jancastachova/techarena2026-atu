#!/usr/bin/env bash
# act_d screen (2026-10-31, lab/last_event_view.py): SET40 + time since the last event / recent visits / no-click streak,
# paired vs SET40. TEST_DAY=6/5, then 4 to confirm.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-31_lab_act_d_d$T.txt
source experiments/lab/bundle_args.sh
BASEP=data/cache/lab_preds_40_d$T.npz; [ $T = 4 ] && BASEP=data/cache/lab_preds_audit_set40_d4.npz
{
$PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" --base-extra act_d=data/cache/43_act_d_c7.npz --save-preds data/cache/lab_preds_act_d_d$T.npz | grep "base (="
$PY experiments/lab/paired_ci.py "$BASEP:base (= the best member)" "data/cache/lab_preds_act_d_d$T.npz:base (= the best member)" | grep -E "plain|weighted"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
