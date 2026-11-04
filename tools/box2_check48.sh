#!/usr/bin/env bash
# 2026-11-03 fourth box, last job: does 48's composition hold on the CURRENT feature set (SET45), days 6 and 5? Members via the
# fast path (lab/lgb_export.py on the exports): binary LightGBM, lambdarank G500 2200 trees; choice SP from tools/box2_sq.sh.
# BASE = binary .5 + SP .5; 48-like = binary .375 + lambdarank .125 + SP .5. Then the box is destroyed by the caller.
set -uo pipefail
cd /workspace/t; PY=.venv/bin/python; L=experiments/logs/2026-11-03_box2; C=data/cache/n1
for T in 6 5; do
  ( $PY experiments/lab/lgb_export.py $C/exp45_d$T.npz --threads 8 --out $C/d${T}_lgbX.npz > $L/lgbX_d$T.txt 2>&1 ) &
  ( $PY experiments/lab/lgb_export.py $C/exp45_d$T.npz --threads 16 --rank 500 --trees 2200 --out $C/d${T}_rankX.npz > $L/rankX_d$T.txt 2>&1 ) &
done
wait
for T in 6 5; do
  $PY experiments/lab/ens_mix.py $C/d${T}_base48.npz b=$C/d${T}_lgbX.npz:0.5 c=$C/d${T}_chSP.npz:0.5 > /dev/null
  $PY experiments/lab/ens_mix.py $C/d${T}_try48.npz b=$C/d${T}_lgbX.npz:0.375 r=$C/d${T}_rankX.npz:0.125 c=$C/d${T}_chSP.npz:0.5 > /dev/null
  { echo "### day $T: binary $(cat $L/lgbX_d$T.txt | grep lgb_export)"; echo "    rank $(grep lgb_export $L/rankX_d$T.txt)"
    echo "## day $T: binary .5 + SP .5 (A) vs binary .375 + lambdarank .125 + SP .5 (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_base48.npz:base (= the best member)" "$C/d${T}_try48.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  } > $L/check48_d$T.txt &
done
wait; cat $L/check48_d*.txt; echo "check48 done"
