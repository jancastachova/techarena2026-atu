#!/usr/bin/env bash
# 2026-11-03 fourth box (EPYC 9654, 48 cores, RTX 6000 Ada): the FINAL file's extra seeds. 45 builds (trees + _45 choice) for
# seeds 6-11, three at a time at 16 threads, while the GPU builds the SP and D45 choice members for the same seeds; then
# 47_spec_choice/build.sh swaps them in at home (47s12).
set -uo pipefail
cd /workspace/t
( for B in "6 7 8" "9 10 11"; do NTHREADS=16 bash experiments/45_aud40/build.sh "$B"; done; echo "trees done $(date)" ) &
( bash tools/box_choiceSP_build.sh "6 7 8 9 10 11"; echo "SP done $(date)" ) &
( bash tools/box_choiceD45_build.sh "6 7 8 9 10 11"; echo "D45 done $(date)" ) &
wait
ls submissions/45_aud40_s*.csv data/cache/24_choice_full_*_test.npy; echo "final done $(date)"
