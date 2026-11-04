#!/usr/bin/env bash
# One box's share of the family x seed x cutoff fits: bash run_box.sh <box index 0-2> <n boxes> "<seeds>" "<cutoffs>"
cd /workspace/t; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
BOX=$1; NB=$2; SEEDS=$3; CUTS=$4
declare -A FAM
FAM[F1]="--cats 8 --coupling 1 --norm-strength 10 --xprec 1 --y0 1.5"
FAM[F2]="--cats 8 --coupling 3 --norm-strength 30 --xprec 1 --y0 2"
FAM[F3]="--cats 8 --coupling 3 --y0 1.5"
FAM[F4]="--cats 16 --coupling 3 --y0 1.5"
i=0
for C in $CUTS; do for S in $SEEDS; do for F in ${FAMLIST:-F1 F2 F3 F4}; do
  i=$((i+1)); [ $((i % NB)) -eq $BOX ] || continue
  O=data/cache/49/${F}_s${S}_c${C}.npz; [ -f $O ] && continue
  E=99; [ $C = 3 ] && E=4
  python experiments/49_bayes_choice/bayes.py --cutoff $C --end $E --seed $S --log-every 100 --map 60 --hyper-start 99999 \
      --sweeps 640 --burn 60 --thin 1 ${FAM[$F]} --out $O > experiments/logs/49/${F}_s${S}_c${C}.txt 2>&1
done; done; done
echo box $BOX done
