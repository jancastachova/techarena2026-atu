#!/usr/bin/env bash
# 2026-11-03 box: item-item CF with cosine / shrunk cosine / BM25 similarity (lab/item_cf.py), then each as an extra group
# on SET45 LightGBM, days 6 and 4 (lab/extra_member.py prints base and base + each extra).
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-03_box_cf; mkdir -p $L
source experiments/lab/bundle_args.sh
SET45=(); for x in "${SET40[@]}"; do [ "$x" = "soft_aud=data/cache/34c_aud_c7.npz" ] && x="soft_aud=data/cache/40c_aud_c7.npz"; SET45+=("$x"); done
$PY experiments/lab/item_cf.py --out-prefix data/cache/48cf > $L/item_cf.txt 2>&1
for T in 6 4; do
  NTHREADS=16 TEST_DAY=$T $PY experiments/lab/extra_member.py --seq "$SEQ" "${SET45[@]}" \
      --extra cf_cos=data/cache/48cf_cos_c7.npz --extra cf_coss=data/cache/48cf_coss_c7.npz --extra cf_bm25=data/cache/48cf_bm25_c7.npz \
      > $L/lgb_d$T.txt 2>&1 &
done
wait
grep -h "AUC\|base (=\|cf_" $L/*.txt | cut -c1-230; echo "cf done"
