#!/usr/bin/env bash
# 2026-11-03 box: a second spectral view from the 41 vectors (decayed-lr MF, 13000 steps) added to the spectral choice member
# (SP41 = SET45 + 47_spec + 47_spec41). Judged as H3 with SP -> SP41 vs H3, days 6/5/4/3.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
$PY experiments/lab/spectral_view.py --family 41 --out data/cache/47_spec41_c7.npz > experiments/logs/2026-11-03_lab_spectral_view41.txt 2>&1
for T in 6 5 4 3; do
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --add data/cache/47_spec41_c7.npz \
      --device cuda --seeds 3 --out $C/d${T}_chSP41.npz > $L/chSP41_d$T.txt 2>&1
  $PY experiments/lab/ens_mix.py $C/d${T}_ensH341.npz lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2 \
      a=$C/d${T}_chSP41.npz:0.2 b=$C/d${T}_chD.npz:0.2 > /dev/null
  ( echo "### day $T choice SP41: $(grep 'choice model ' $L/chSP41_d$T.txt)  (SP: $(grep 'choice model ' $L/chSP_d$T.txt | grep -o 'eval-weighted [0-9.]*'))"
    echo "## day $T: H3 (A) vs H3 with SP -> SP41 (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensH3.npz:base (= the best member)" "$C/d${T}_ensH341.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  ) > $L/sp41_d$T.txt &
done
wait
cat $L/sp41_d*.txt > $L/compare_sp41.txt; cat $L/compare_sp41.txt
