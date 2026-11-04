#!/usr/bin/env bash
# Batch 3 (2026-11-01, vast.ai box): the neural CHOICE member re-tuned on SET40 (its build uses SET40 since 40; tuned only on
# the lean set before). 4 configs x days 6/5/4, 2 seeds each, 6 jobs at a time (8 torch threads each). Judged as the build
# uses it: equal rank average of SET40 LightGBM + W2 + choice(config), paired vs the same with the default config.
#   C0 default (256,128, dropout 0.1)   C1 512,256,128 dropout 0.2   C2 256,128 dropout 0.3   C3 1024,512 dropout 0.3 lr 5e-4
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-01_box_batch3; mkdir -p $L
run() { local C=$1 T=$2; shift 2
  $PY experiments/lab/choice_model.py data/cache/lab_export_set40_d$T.npz --seeds 2 --out data/cache/lab_preds_ch${C}_d$T.npz "$@" \
      > $L/ch${C}_d$T.txt 2>&1; }
export -f run; export PY L
{
for T in 6 5 4; do
  echo "C0 $T"; echo "C1 $T --hidden 512,256,128 --drop 0.2"; echo "C2 $T --drop 0.3"; echo "C3 $T --hidden 1024,512 --drop 0.3 --lr 5e-4 --epochs 20"
done
} | xargs -P 6 -L 1 bash -c 'run "$@"' _
w2() { [ $1 = 4 ] && echo data/cache/lab_preds_div_W2_d4.npz || echo data/cache/lab_preds_div_W2_d$1.npz; }
for T in 6 5 4; do
  B=data/cache/lab_preds_boxbase_d$T.npz
  $PY experiments/lab/avg_n.py data/cache/lab_preds_ens2_d$T.npz $B $(w2 $T)
  for C in C0 C1 C2 C3; do $PY experiments/lab/avg_n.py data/cache/lab_preds_ens3${C}_d$T.npz $B $(w2 $T) data/cache/lab_preds_ch${C}_d$T.npz; done
  echo "### day $T: choice alone (C0) vs box base" >> $L/paired.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "$B:base (= the best member)" "data/cache/lab_preds_chC0_d$T.npz:base (= the best member)" 2>&1 | grep -E "eval-w|gap-w" >> $L/paired.txt
  echo "### day $T: LGB+W2+choice(C0) vs LGB+W2" >> $L/paired.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_ens2_d$T.npz:base (= the best member)" "data/cache/lab_preds_ens3C0_d$T.npz:base (= the best member)" 2>&1 | grep -E "eval-w|gap-w" >> $L/paired.txt
  for C in C1 C2 C3; do
    echo "### day $T: ens3 with $C vs with C0" >> $L/paired.txt
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_ens3C0_d$T.npz:base (= the best member)" "data/cache/lab_preds_ens3${C}_d$T.npz:base (= the best member)" 2>&1 | grep -E "eval-w|gap-w" >> $L/paired.txt
  done
done
cat $L/paired.txt
