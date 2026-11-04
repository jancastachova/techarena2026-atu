#!/usr/bin/env bash
# 2026-11-03 box: CHOICE member on SET45 + the RAW spectral view (47_specraw: per-direction products of raw vectors, the dot
# product's decomposition), vs choice S and choice SP inside the 45 ensemble, days 6/5/4/3.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
for T in 6 5 4 3; do
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_specraw_c7.npz --device cuda --seeds 3 \
      --out $C/d${T}_chSR.npz > $L/chSR_d$T.txt 2>&1
  $PY experiments/lab/ens_mix.py $C/d${T}_ensSR.npz lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2 ch=$C/d${T}_chSR.npz:0.4 > /dev/null
  { echo "### day $T choice SR: $(grep 'choice model ' $L/chSR_d$T.txt)"
    echo "## day $T ensemble: choice S (A) vs choice + raw spectral (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensS.npz:base (= the best member)" "$C/d${T}_ensSR.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted"
  } >> $L/compare_specraw.txt
done
