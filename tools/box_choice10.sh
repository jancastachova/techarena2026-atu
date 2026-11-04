#!/usr/bin/env bash
# Choice member, stronger (2026-11-01, box GPU): setting C3 (1024,512, dropout 0.3, lr 5e-4, <= 20 epochs) with 10 seeds, days
# 6/5/4 on the GPU at once (67 s per seed vs ~20 min on the CPU). Judged in the ensemble as the next build would use it:
# LightGBM 0.4 : W2 0.2 : choice 0.4, choice = C3 x10 vs C0 x2 (batch 3); also choice alone.
set -uo pipefail
PY=.venv/bin/python; L=experiments/logs/2026-11-01_box_choicegpu; mkdir -p $L
for T in 6 5 4; do
  $PY experiments/lab/choice_model.py data/cache/lab_export_set40_d$T.npz --seeds 10 --device cuda --hidden 1024,512 --drop 0.3 \
      --lr 5e-4 --epochs 20 --out data/cache/lab_preds_chC3x10_d$T.npz > $L/c3x10_d$T.txt 2>&1 &
done
wait
cat > /tmp/w.py <<'PY'
import sys, numpy as np
from scipy.stats import rankdata
T, ch, out = sys.argv[1:4]; K = "base (= the best member)|B"
Z = [np.load(f) for f in (f"data/cache/lab_preds_boxbase_d{T}.npz", f"data/cache/lab_preds_div_W2_d{T}.npz", ch)]
r = Z[0]["rows_B"]; R = [rankdata(z[K]) / len(r) for z in Z]
np.savez(out, rows_B=r, **{K: 0.4 * R[0] + 0.2 * R[1] + 0.4 * R[2]})
PY
for T in 6 5 4; do
  $PY /tmp/w.py $T data/cache/lab_preds_chC0_d$T.npz data/cache/lab_preds_ensw_C0_d$T.npz
  $PY /tmp/w.py $T data/cache/lab_preds_chC3x10_d$T.npz data/cache/lab_preds_ensw_C3x10_d$T.npz
  echo "### day $T: choice C3x10 alone: $(grep 'choice model ' $L/c3x10_d$T.txt)" >> $L/compare.txt
  echo "### day $T: ensemble (choice 0.4) C3x10 vs C0x2" >> $L/compare.txt
  TEST_DAY=$T $PY experiments/lab/paired_ci.py "data/cache/lab_preds_ensw_C0_d$T.npz:base (= the best member)" \
      "data/cache/lab_preds_ensw_C3x10_d$T.npz:base (= the best member)" 2>&1 | grep -E "plain|weighted" >> $L/compare.txt
done
cat $L/compare.txt
