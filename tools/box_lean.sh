#!/usr/bin/env bash
# 2026-11-03 box: LEAN compositions of 47 (choice SP + D at 0.25 each) vs 47 (trees lgb / xgb / W2 at 1/6 each), days 6-3:
# L1 drop xgb (lgb 0.25, W2 0.25), L2 drop W2 (lgb 0.25, xgb 0.25), L3 lgb alone 0.5.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
for T in 6 5 4 3; do
  CH="a=$C/d${T}_chSP.npz:0.25 b=$C/d${T}_chD.npz:0.25"
  $PY experiments/lab/ens_mix.py $C/d${T}_ens47.npz lgb=$C/d${T}_lgb.npz:0.16667 xgb=$C/d${T}_xgb.npz:0.16667 w2=$C/d${T}_w2.npz:0.16667 $CH > /dev/null
  $PY experiments/lab/ens_mix.py $C/d${T}_ensL1.npz lgb=$C/d${T}_lgb.npz:0.25 w2=$C/d${T}_w2.npz:0.25 $CH > /dev/null
  $PY experiments/lab/ens_mix.py $C/d${T}_ensL2.npz lgb=$C/d${T}_lgb.npz:0.25 xgb=$C/d${T}_xgb.npz:0.25 $CH > /dev/null
  $PY experiments/lab/ens_mix.py $C/d${T}_ensL3.npz lgb=$C/d${T}_lgb.npz:0.5 $CH > /dev/null
  for v in L1 L2 L3; do
    ( echo "## day $T: 47 (A) vs $v (B)"
      TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ens47.npz:base (= the best member)" "$C/d${T}_ens$v.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
    ) > $L/lean_d${T}_$v.txt &
  done
  ( echo "## day $T: 45 composition (A) vs 47 (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensS.npz:base (= the best member)" "$C/d${T}_ens47.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  ) > $L/lean_d${T}_47.txt &
done
wait
cat $L/lean_d*.txt > $L/compare_lean.txt; cat $L/compare_lean.txt
