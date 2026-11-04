#!/usr/bin/env bash
# Batch 1 on the vast.ai box (2026-11-01): 8 jobs side by side, 24 threads each (NTHREADS), after tools/box_verify.sh.
#   lr01_d{6,5,4}   SET40 LightGBM at learning rate 0.01 (lab default 0.02), paired vs SET40 -- the one tree setting never tried
#   export_d{6,5,4} SET40 feature matrices + splits for the neural ranker (lab/extra_member.py EXPORT=)
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24
source experiments/lab/bundle_args.sh
L=experiments/logs/2026-11-01_box_batch1; mkdir -p $L
base_of() { [ $1 = 4 ] && echo data/cache/lab_preds_audit_set40_d4.npz || echo data/cache/lab_preds_40_d$1.npz; }
for T in 6 5 4; do
  ( TEST_DAY=$T LGBP='{"learning_rate": 0.01}' $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" \
        --save-preds data/cache/lab_preds_lr01_d$T.npz 2>&1 | grep -E "base \(=" > $L/lr01_d$T.txt
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$(base_of $T):base (= the best member)" "data/cache/lab_preds_lr01_d$T.npz:base (= the best member)" 2>&1 \
        | grep -E "plain|weighted" >> $L/lr01_d$T.txt ) &
  ( TEST_DAY=$T EXPORT=data/cache/lab_export_set40_d$T.npz $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" \
        > $L/export_d$T.txt 2>&1 ) &
done
wait
for f in $L/*.txt; do echo "## $f"; cat $f | tail -5; done
