#!/usr/bin/env bash
# Choice model variants on the lean export (lab/choice_model.py): slate attention, 3 seeds. TEST_DAY=5 for the holdout.
set -euo pipefail
T=${TEST_DAY:-6}
if [ "$T" = 6 ]; then BL=(--blend data/cache/lab_preds_lean0_d6.npz); else BL=(--blend data/cache/lab_preds_lean_d5.npz --blend-key "top 15 groups (169 cols)"); fi
P=(.venv/bin/python experiments/lab/choice_model.py data/cache/lab_export_lean_d$T.npz "${BL[@]}")
{
echo "### attn, 1 seed"; "${P[@]}" --attn --out data/cache/lab_preds_choice_attn_d$T.npz
echo "### plain, 3 seeds"; "${P[@]}" --seeds 3 --out data/cache/lab_preds_choice_s3_d$T.npz
} 2>&1 | grep -v "epoch" | tee experiments/logs/2026-10-30_lab_choice_v2_d$T.txt
