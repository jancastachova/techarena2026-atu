#!/usr/bin/env bash
# (h) = 45 set + the coherence / typicality view (lab/coherence_view.py -> 46_coh), vs the 45 set re-run on the same machine,
# days 6/5/4 (2026-11-02, second vast.ai box: 96 threads, 109 GB). 6 jobs x 16 threads.
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=16
source experiments/lab/bundle_args.sh
A=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; A+=("$x"); done
H=("${A[@]}" --base-extra coh=data/cache/46_coh_c7.npz)
L=experiments/logs/2026-11-02_box_coh; mkdir -p $L
for T in 6 5 4; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${A[@]}" --save-preds data/cache/lab_preds_b2base_d$T.npz 2>&1 | grep "base (=" > $L/base_d$T.txt ) &
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${H[@]}" --save-preds data/cache/lab_preds_coh_d$T.npz 2>&1 | grep "base (=" > $L/coh_d$T.txt ) &
done
wait
for T in 6 5 4; do
  echo "### day $T: (h) coherence vs 45 set (same box)" >> $L/compare.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_b2base_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_coh_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done
echo "### day 6: this box's 45 baseline vs the first box's" >> $L/compare.txt
TEST_DAY=6 $PY experiments/lab/paired_ci.py "data/cache/box/lab/lab_preds_aud40a_d6.npz:base (= the best member)" \
    "data/cache/lab_preds_b2base_d6.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
cat $L/compare.txt
