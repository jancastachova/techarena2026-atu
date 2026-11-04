#!/usr/bin/env bash
# Batch 2 on the vast.ai box (2026-11-01), 24 threads per job:
#   base_d{6,5,4}    SET40 at lr 0.02 computed ON THE BOX (the Mac baselines differ by thread-count noise, about +-0.0001)
#   w2lr01_d{6,5,4}  the shallow W2 member (42) at lr 0.01
# Then paired comparisons box vs box: lr 0.01 vs base, and W2 lr 0.01 vs base.
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24
source experiments/lab/bundle_args.sh
L=experiments/logs/2026-11-01_box_batch2; mkdir -p $L
W2='{"num_leaves": 15, "min_data_in_leaf": 2000, "feature_fraction": 0.3, "learning_rate": 0.01}'
for T in 6 5 4; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" --save-preds data/cache/lab_preds_boxbase_d$T.npz 2>&1 | grep -E "base \(=" > $L/base_d$T.txt ) &
  ( TEST_DAY=$T LGBP="$W2" $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET40[@]}" --save-preds data/cache/lab_preds_w2lr01_d$T.npz 2>&1 | grep -E "base \(=" > $L/w2lr01_d$T.txt ) &
done
wait
while pgrep -f "lab_preds_lr01_d" > /dev/null; do sleep 20; done       # batch 1's lr 0.01 runs
for T in 6 5 4; do
  for C in lr01 w2lr01; do
    echo "### day $T: $C vs box base" >> $L/paired.txt
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_boxbase_d$T.npz:base (= the best member)" \
        "data/cache/lab_preds_${C}_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/paired.txt
  done
  $PY experiments/lab/avg_preds.py data/cache/lab_preds_lr01w2avg_d$T.npz data/cache/lab_preds_lr01_d$T.npz data/cache/lab_preds_w2lr01_d$T.npz
  echo "### day $T: avg(lr01, w2lr01) vs box base" >> $L/paired.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_boxbase_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_lr01w2avg_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/paired.txt
done
cat $L/*.txt
