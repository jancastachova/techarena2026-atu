#!/usr/bin/env bash
# 2026-11-03 box: CHOICE member on SET45 + spectral + per-direction taste SPREAD (spectral_view.py --var -> 47_specvar) = SPV.
# Judged as H3 with SP replaced by SPV vs H3, days 6/5/4/3.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
$PY experiments/lab/spectral_view.py --var --out data/cache/47_specvar_c7.npz > experiments/logs/2026-11-03_lab_spectral_var.txt 2>&1
for T in 6 5 4 3; do
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --add data/cache/47_specvar_c7.npz \
      --device cuda --seeds 3 --out $C/d${T}_chSPV.npz > $L/chSPV_d$T.txt 2>&1
  $PY experiments/lab/ens_mix.py $C/d${T}_ensH3V.npz lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2 \
      a=$C/d${T}_chSPV.npz:0.2 b=$C/d${T}_chD.npz:0.2 > /dev/null
  ( echo "### day $T choice SPV: $(grep 'choice model ' $L/chSPV_d$T.txt)  (SP: $(grep 'choice model ' $L/chSP_d$T.txt | grep -o 'eval-weighted [0-9.]*'))"
    echo "## day $T: H3 (A) vs H3 with SP -> SPV (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensH3.npz:base (= the best member)" "$C/d${T}_ensH3V.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  ) > $L/spv_d$T.txt &
done
wait
cat $L/spv_d*.txt > $L/compare_spv.txt; cat $L/compare_spv.txt
