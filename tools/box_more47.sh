#!/usr/bin/env bash
# 2026-11-03 box: seeds 6-11 for experiment 47 (fixed average, no new settings). Per seed: the 45 build (trees + _45 choice,
# 3 seeds in parallel), then the SP and D45 choice members on the GPU; 47_spec_choice/build.sh then swaps them in at home.
set -uo pipefail
cd /workspace/t
for B in "6 7 8" "9 10 11"; do bash experiments/45_aud40/build.sh "$B"; done
echo "trees done $(date)"
bash tools/box_choiceSP_build.sh "6 7 8 9 10 11" &
bash tools/box_choiceD45_build.sh "6 7 8 9 10 11" &
wait
echo "all done $(date)"
