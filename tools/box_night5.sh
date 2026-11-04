#!/usr/bin/env bash
# 2026-11-03 box: CHOICE member on SET45 + spectral view + SPECTRAL AUDIENCE view (lab/spectral_aud.py -> 47_specaud) = SPA.
# Judged as H3 with SP replaced by SPA (choice SPA 0.2 + D 0.2, trees 0.2 each) vs H3, days 6/5/4/3.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
for T in 6 5 4 3; do
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --add data/cache/47_specaud_c7.npz \
      --device cuda --seeds 3 --out $C/d${T}_chSPA.npz > $L/chSPA_d$T.txt 2>&1
  $PY experiments/lab/ens_mix.py $C/d${T}_ensH3A.npz lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2 \
      a=$C/d${T}_chSPA.npz:0.2 b=$C/d${T}_chD.npz:0.2 > /dev/null
  ( echo "### day $T choice SPA: $(grep 'choice model ' $L/chSPA_d$T.txt)  (SP: $(grep 'choice model ' $L/chSP_d$T.txt | grep -o 'eval-weighted [0-9.]*'))"
    echo "## day $T: H3 (A) vs H3 with SPA (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensH3.npz:base (= the best member)" "$C/d${T}_ensH3A.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  ) > $L/spa_d$T.txt &
done
wait
cat $L/spa_d*.txt > $L/compare_spa.txt; cat $L/compare_spa.txt
