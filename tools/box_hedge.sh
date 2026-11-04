#!/usr/bin/env bash
# 2026-11-03 box: choice-member mixes inside the 45 ensemble (choice total 0.40), vs choice S alone, days 6/5/4/3:
# H1 = S 0.2 + SP 0.2, H2 = S 0.2 + D 0.2, H3 = SP 0.2 + D 0.2, H4 = S + SP + D (0.133 each).
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
for T in 6 5 4 3; do
  M="lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2"
  $PY experiments/lab/ens_mix.py $C/d${T}_ensH1.npz $M a=$C/d${T}_chS.npz:0.2 b=$C/d${T}_chSP.npz:0.2 > /dev/null
  $PY experiments/lab/ens_mix.py $C/d${T}_ensH2.npz $M a=$C/d${T}_chS.npz:0.2 b=$C/d${T}_chD.npz:0.2 > /dev/null
  $PY experiments/lab/ens_mix.py $C/d${T}_ensH3.npz $M a=$C/d${T}_chSP.npz:0.2 b=$C/d${T}_chD.npz:0.2 > /dev/null
  $PY experiments/lab/ens_mix.py $C/d${T}_ensH4.npz $M a=$C/d${T}_chS.npz:0.1333 b=$C/d${T}_chSP.npz:0.1333 c=$C/d${T}_chD.npz:0.1334 > /dev/null
  for h in H1 H2 H3 H4; do
    ( echo "## day $T ensemble: choice S (A) vs $h (B)"
      TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensS.npz:base (= the best member)" "$C/d${T}_ens$h.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
    ) > $L/hedge_d${T}_$h.txt &
  done
done
wait
cat $L/hedge_d*_H*.txt > $L/compare_hedge.txt; cat $L/compare_hedge.txt
