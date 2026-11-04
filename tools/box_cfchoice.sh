#!/usr/bin/env bash
# 2026-11-03 box GPU: the spectral choice member (SP) with the three item-item CF caches added (48cf cos / coss / bm25),
# 47 with SP -> SPCF, days 6/5/4/3, paired eval- / gap-weighted.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_cf; C=data/cache/n1
for T in 6 5 4 3; do
  $PY experiments/lab/choice_model.py $C/exp45_d$T.npz --add data/cache/47_spec_c7.npz --add data/cache/48cf_cos_c7.npz \
      --add data/cache/48cf_coss_c7.npz --add data/cache/48cf_bm25_c7.npz --device cuda --seeds 3 --out $C/d${T}_chSPCF.npz > $L/chSPCF_d$T.txt 2>&1
  $PY experiments/lab/ens_mix.py $C/d${T}_ens47CF.npz lgb=$C/d${T}_lgb.npz:0.16667 xgb=$C/d${T}_xgb.npz:0.16667 \
      w2=$C/d${T}_w2.npz:0.16667 a=$C/d${T}_chSPCF.npz:0.25 b=$C/d${T}_chD.npz:0.25 > /dev/null
  { echo "### day $T SPCF $(grep 'choice model ' $L/chSPCF_d$T.txt) (SP: $(grep 'choice model ' experiments/logs/2026-11-03_box_night1/chSP_d$T.txt | grep -o 'eval-weighted [0-9.]*'))"
    echo "## day $T: 47 (A) vs 47 with SP -> SPCF (B)"
    TEST_DAY=$T $PY experiments/lab/paired_ci.py "$C/d${T}_ens47.npz:base (= the best member)" "$C/d${T}_ens47CF.npz:base (= the best member)" 2>&1 | grep -E "eval-weighted|gap-weighted"
  } > $L/cfchoice_d$T.txt &
done
wait; cat $L/cfchoice_d*.txt; echo "cfchoice done"
