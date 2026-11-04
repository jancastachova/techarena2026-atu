#!/usr/bin/env bash
# 2026-11-03 box: a THIRD choice member for diversity: SET45 + spectral with slate self-attention (choice_model.py --attn) =
# SPT. Judged as H3 + SPT (choice SP, D, SPT at 0.4/3 each, trees 0.2 each) vs H3, days 6/5/4/3.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
for T in 6 5 4 3; do
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --attn \
      --device cuda --seeds 3 --out $C/d${T}_chSPT.npz > $L/chSPT_d$T.txt 2>&1
  $PY experiments/lab/ens_mix.py $C/d${T}_ensH3T.npz lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2 \
      a=$C/d${T}_chSP.npz:0.1333 b=$C/d${T}_chD.npz:0.1333 c=$C/d${T}_chSPT.npz:0.1334 > /dev/null
  ( echo "### day $T choice SPT: $(grep 'choice model ' $L/chSPT_d$T.txt)"
    echo "## day $T: H3 (A) vs H3 + SPT (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensH3.npz:base (= the best member)" "$C/d${T}_ensH3T.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  ) > $L/spt_d$T.txt &
done
wait
cat $L/spt_d*.txt > $L/compare_spt.txt; cat $L/compare_spt.txt
