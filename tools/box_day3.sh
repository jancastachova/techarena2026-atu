#!/usr/bin/env bash
# Day 3, the never-used split, for the frozen shortlist (2026-11-01): members on SET40 trained in parallel, then whole ensembles
# A (40 config) / B (42 config) / C (42 with choice 0.40) composed (lab/ens_compose.py) and compared pairwise.
set -uo pipefail
PY=.venv/bin/python; export TEST_DAY=3 NTHREADS=24
source experiments/lab/bundle_args.sh
L=experiments/logs/2026-11-01_box_day3; mkdir -p $L
X() { $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" "$@"; }
( X --save-preds data/cache/lab_preds_d3_lgb.npz 2>&1 | grep "base (=" > $L/lgb.txt ) &
( LGBP='{"num_leaves": 15, "min_data_in_leaf": 2000, "feature_fraction": 0.3}' X --save-preds data/cache/lab_preds_d3_w2.npz 2>&1 | grep "base (=" > $L/w2.txt ) &
( MODEL=xgb XGB_DEVICE=cuda X --save-preds data/cache/lab_preds_d3_xgb.npz 2>&1 | grep -E "base \(=|Error|Trace" > $L/xgb.txt ) &
( EXPORT=data/cache/lab_export_set40_d3.npz X > $L/export.txt 2>&1
  $PY experiments/lab/choice_model.py data/cache/lab_export_set40_d3.npz --seeds 3 --out data/cache/lab_preds_d3_choice.npz > $L/choice.txt 2>&1 ) &
wait
$PY experiments/lab/ens_compose.py 3 > $L/compose.txt 2>&1
for pair in "A B" "B C" "A C"; do set -- $pair
  echo "### day 3: ens$1 (A) vs ens$2 (B)" >> $L/compare.txt
  $PY experiments/lab/paired_ci.py "data/cache/lab_preds_d3_ens$1.npz:base (= the best member)" "data/cache/lab_preds_d3_ens$2.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done
cat $L/lgb.txt $L/w2.txt $L/xgb.txt; grep "choice model " $L/choice.txt; cat $L/compose.txt $L/compare.txt
