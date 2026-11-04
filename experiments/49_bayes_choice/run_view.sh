#!/usr/bin/env bash
# All-day point-estimate fits, cutoffs 2-6: bash run_view.sh <box 0-2>
cd /workspace/t; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
BOX=$1; i=0
for C in 6 5 4 3 2; do for V in gau c16; do
  i=$((i+1)); [ $((i % 3)) -eq $BOX ] || continue
  [ $V = gau ] && FL="--cats 8" || FL="--cats 16 --norm-strength 10 --xprec 1"
  echo "python experiments/49_bayes_choice/bayes.py --cutoff $C --all-day --log-every 59 --map 60 --hyper-start 99999 --sweeps 60 --burn 59 --thin 1 --coupling 1 --y0 1.5 $FL --out data/cache/49/V${V}_c$C.npz > experiments/logs/49/V${V}_c$C.txt 2>&1"
done; done > /tmp/vcmds.txt
xargs -P 2 -I{} bash -c "{}" < /tmp/vcmds.txt
echo view box $BOX finished $(ls data/cache/49 | grep -c "^V")
