#!/usr/bin/env bash
# 2026-11-03 box: CHOICE member on D45 + spectral view (DSP). Judged as H3 with D replaced by DSP (SP 0.2 + DSP 0.2) vs H3.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_night1; C=data/cache/n1
for T in 6 5 4 3; do
  $PY experiments/lab/choice_model.py $C/expD_d$T.npz --add data/cache/47_spec_c7.npz \
      --device cuda --seeds 3 --out $C/d${T}_chDSP.npz > $L/chDSP_d$T.txt 2>&1
  $PY experiments/lab/ens_mix.py $C/d${T}_ensH3D.npz lgb=$C/d${T}_lgb.npz:0.2 xgb=$C/d${T}_xgb.npz:0.2 w2=$C/d${T}_w2.npz:0.2 \
      a=$C/d${T}_chSP.npz:0.2 b=$C/d${T}_chDSP.npz:0.2 > /dev/null
  ( echo "### day $T choice DSP: $(grep 'choice model ' $L/chDSP_d$T.txt)  (SP: $(grep 'choice model ' $L/chSP_d$T.txt | grep -o 'eval-weighted [0-9.]*'))"
    echo "## day $T: H3 (A) vs H3 with D -> DSP (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ensH3.npz:base (= the best member)" "$C/d${T}_ensH3D.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  ) > $L/dsp_d$T.txt &
done
wait
cat $L/dsp_d*.txt > $L/compare_dsp.txt; cat $L/compare_dsp.txt
