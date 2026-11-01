#!/usr/bin/env bash
# Lists the data files a remote machine (vast.ai) needs to run the current lab screens, the SET40/42 builds and new MF work,
# without train.csv (the base features are cached) and without the ~48 GB of older caches.
#     bash tools/vast_manifest.sh > /tmp/vast_files.txt
#     rsync -avP --files-from=/tmp/vast_files.txt . root@HOST:/workspace/techarena2026-atu/ -e "ssh -p PORT"
set -euo pipefail
source experiments/lab/bundle_args.sh
{
  echo data/evaluation.csv
  echo data/cache/13_base_c7_9c47a117c762.npz
  for f in 13_mf_c7_bb1fcb0a9c89 15_mf_c7_46f30c940a6a 20_mf_c7_c6b710c438b5; do echo data/cache/$f.npz; done
  echo "$SEQ" | tr ',' '\n'
  for x in "${SET40[@]}" "${SET38[@]}"; do [ "$x" = "--base-extra" ] && continue; echo "${x#*=}" | tr ',' '\n'; done
  echo data/cache/43_act_d_c7.npz
  ls data/cache/lab_preds_40_d*.npz data/cache/lab_preds_audit_set40_d4.npz data/cache/lab_preds_lean0_d*.npz
  ls data/cache/40_vectors_c7_s*.npz data/cache/41_vectors_c7_s*.npz
  ls data/cache/lab_export_lean_d*.npz 2>/dev/null || true
} | sort -u
