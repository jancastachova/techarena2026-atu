#!/usr/bin/env bash
# Choice model settings on the lean export, day 6, 2 seeds each (lab/choice_model.py); baseline 256-128 / lr 1e-3 / drop 0.1.
set -euo pipefail
P=(.venv/bin/python experiments/lab/choice_model.py data/cache/lab_export_lean_d6.npz --seeds 2)
{
for v in "base|" "wide|--hidden 512,256" "deep|--hidden 512,256,128" "lr5e-4|--lr 5e-4 --epochs 20" "drop0.2|--drop 0.2 --hidden 512,256"; do
  n=${v%%|*}; o=${v#*|}; echo "### $n $o"
  "${P[@]}" $o --out data/cache/lab_preds_choice_sw_${n}_d6.npz
done
} 2>&1 | grep -v "epoch" | tee experiments/logs/2026-10-30_lab_choice_sweep_d6.txt
