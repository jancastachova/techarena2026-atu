#!/usr/bin/env bash
# Stronger soft-projected audience views for 45 (2026-11-01): 40c_aud used 5 of the 10 40-vector seeds. Rebuilt from all 10
# (40d), from the 10 lr-decayed 41 seeds (41c), and from all 20 (4041c); same recipe (--cosine --unit-clickers --soft-proj 20).
set -euo pipefail
PY=.venv/bin/python; A="experiments/26_audience/audience.py --cosine --unit-clickers --soft-proj 20"
V40=$(for s in 0 1 2 3 4 5 6 7 8 9; do printf "data/cache/40_vectors_c7_s%s.npz," $s; done); V40=${V40%,}
V41=$(for s in 0 1 2 3 4 5 6 7 8 9; do printf "data/cache/41_vectors_c7_s%s.npz," $s; done); V41=${V41%,}
$PY $A --vectors $V40 --out-prefix data/cache/40d > experiments/logs/2026-11-01_aud_40d.txt 2>&1 &
$PY $A --vectors $V41 --out-prefix data/cache/41c > experiments/logs/2026-11-01_aud_41c.txt 2>&1 &
wait
$PY $A --vectors $V40,$V41 --out-prefix data/cache/4041c > experiments/logs/2026-11-01_aud_4041c.txt 2>&1
grep -h "AUC\|wrote" experiments/logs/2026-11-01_aud_40d.txt experiments/logs/2026-11-01_aud_41c.txt experiments/logs/2026-11-01_aud_4041c.txt
