#!/usr/bin/env bash
# Experiment 41b screen: the projected view averaged over all 20 vector seeds (40's 10 + 41's 10), swapped into SET40, paired vs SET40.
set -euo pipefail
PY=.venv/bin/python; T=${TEST_DAY:-6}; L=experiments/logs/2026-10-31_41_screen20_d$T.txt
source experiments/lab/bundle_args.sh
V=data/cache/41_cosproj8_20s_c7.npz
[ -f $V ] || $PY experiments/34_cosine/cos_view.py --proj 8 --vectors $(for s in 0 1 2 3 4 5 6 7 8 9; do printf "40_vectors_c7_s%s 41_vectors_c7_s%s " $s $s; done) --out $V
NEW=(); for x in "${SET40[@]}"; do [ "$x" = "cosproj8=data/cache/40_cosproj8_c7.npz" ] && x="cosproj8=$V"; NEW+=("$x"); done
BASEP=data/cache/lab_preds_40_d$T.npz; [ $T = 4 ] && BASEP=data/cache/lab_preds_audit_set40_d4.npz
{
$PY experiments/lab/extra_member.py --seq "$SEQ" "${NEW[@]}" --save-preds data/cache/lab_preds_41_20s_d$T.npz
$PY experiments/lab/paired_ci.py "$BASEP:base (= the best member)" "data/cache/lab_preds_41_20s_d$T.npz:base (= the best member)"
} 2>&1 | grep -v "^\[LightGBM\]" | tee $L
