#!/usr/bin/env bash
# 2026-11-03 box: H3 (choice SP + D, 0.2 each, trees 0.2 each) vs the same with the choice pair at 0.5 / 0.6 in total, days 6-3.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
for T in 6 5 4 3; do
  for w in 0.5 0.6; do
    t=$(python3 -c "print(round((1-$w)/3,4))"); c=$(python3 -c "print($w/2)")
    $PY experiments/lab/ens_mix.py $C/d${T}_ensH3w$w.npz lgb=$C/d${T}_lgb.npz:$t xgb=$C/d${T}_xgb.npz:$t w2=$C/d${T}_w2.npz:$t \
        a=$C/d${T}_chSP.npz:$c b=$C/d${T}_chD.npz:$c > /dev/null
    ( echo "## day $T: H3 (A) vs H3 with choice $w (B)"
      TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensH3.npz:base (= the best member)" "$C/d${T}_ensH3w$w.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
    ) > $L/hedge2_d${T}_$w.txt &
  done
done
wait
cat $L/hedge2_d*.txt > $L/compare_hedge2.txt; cat $L/compare_hedge2.txt
