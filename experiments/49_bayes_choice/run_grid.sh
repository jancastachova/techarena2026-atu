#!/usr/bin/env bash
# Point-estimate variants (fixed-prior alternating Newton, no sampling): bash run_grid.sh <box 0-2> <n boxes> "<cutoffs>" "<names or all>"
cd /workspace/t; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
BOX=$1; NB=$2; CUTS=$3; ONLY=${4:-all}
declare -A G
N1="--cats 8 --coupling 1 --norm-strength 10 --xprec 1 --y0 1.5"
G[K16]="$N1 --K 16"; G[K32]="$N1 --K 32"; G[K12]="$N1 --K 12"
G[pr3]="--cats 8 --coupling 3 --xprec 3 --sd-a 0.15 --y0 2"; G[k12p]="--cats 8 --coupling 3 --xprec 3 --sd-a 0.15 --y0 2 --K 12"
G[pw05]="$N1 --pw 0.5"; G[pw2]="$N1 --pw 2"; G[c16]="--cats 16 --coupling 1 --norm-strength 10 --xprec 1 --y0 1.5"
G[c0]="--cats 0 --norm-strength 10 --xprec 1 --y0 1.5"; G[gau]="--cats 8 --coupling 1 --y0 1.5"; G[y1]="--cats 8 --coupling 1 --norm-strength 10 --xprec 1 --y0 1"
i=0
for n in K16 K32 K12 pr3 k12p pw05 pw2 c16 c0 gau y1; do
  [ "$ONLY" = all ] || [[ " $ONLY " == *" $n "* ]] || continue
  i=$((i+1)); [ $((i % NB)) -eq $BOX ] || continue
  for C in $CUTS; do
    O=data/cache/49/G${n}_c$C.npz; [ -f $O ] && continue
    E=99; [ $C = 3 ] && E=4
    python experiments/49_bayes_choice/bayes.py --cutoff $C --end $E --log-every 59 --map 60 --hyper-start 99999 --sweeps 60 --burn 59 --thin 1 ${G[$n]} --out $O > experiments/logs/49/G${n}_c$C.txt 2>&1
  done
done
echo grid box $BOX done
