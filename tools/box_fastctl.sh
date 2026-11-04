#!/usr/bin/env bash
# 2026-11-03: control for lab/lgb_export.py (day 6, no additions) vs the extra_member.py member d6_lgb (eval-weighted 0.82822
# region); started once the CF LightGBM jobs have finished, timed.
cd /workspace/t
while pgrep -f "[e]xtra_member.py --seq" > /dev/null; do sleep 60; done
date; /usr/bin/time -v .venv/bin/python experiments/lab/lgb_export.py data/cache/n1/exp45_d6.npz --out data/cache/n1/d6_lgbctl.npz 2>&1 | grep -E "lgb_export|Elapsed|Maximum resident"
grep -h "base (=" experiments/logs/2026-11-03_box_night1/lgb_d6.txt | cut -c1-160; date; echo "fastctl done"
