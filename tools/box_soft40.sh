#!/usr/bin/env bash
# (f) = 45 set + the soft-projected cosine view (cos_view.py --proj 8 --soft 20) rebuilt from the 10 40 vectors (cossoft20 <-
# 40_cossoft20; the 34 version was dropped by the lean screen), vs 45 set, days 6/5/4 (2026-11-02).
set -uo pipefail
PY=.venv/bin/python; export NTHREADS=24
source experiments/lab/bundle_args.sh
D=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; D+=("$x"); done; D+=(--base-extra cossoft20=data/cache/40_cossoft20_c7.npz)
L=experiments/logs/2026-11-02_box_soft40; mkdir -p $L
for T in 6 5 4; do
  ( TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${D[@]}" --save-preds data/cache/lab_preds_soft40_d$T.npz 2>&1 | grep "base (=" > $L/d$T.txt ) &
done
wait
for T in 6 5 4; do
  echo "### day $T: (f) vs 45 set" >> $L/compare.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_aud40a_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_soft40_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done
cat $L/compare.txt
