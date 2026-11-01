# Run from the repository root: .venv/bin/python experiments/lab/repeat_position.py
# lab/replica_ppc.py: real repeat clicks favour the MOST RECENT clicks (re-click of the item 1 / 2 / 5 clicks back
# 1.57x / 1.46x / 1.30x the replica's rate). rep_recency measures recency in TIME. Is the bundle-set LightGBM
# calibrated across the repeat's position counted in CLICKS? Half-B predictions saved by lab/extra_member.py
# --save-preds (data/cache/lab_preds_uw1.npz, "base" member).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(ROOT / "data/cache/lab_preds_uw1.npz")
key = [k for k in z.files if k.startswith("base")][0]
pb = z[key]
row_day = np.repeat(fit["day"], 10); rows = np.flatnonzero(row_day >= 2)
rB = rows[z["rows_B"]]                                  # global row ids of half B
ev = rB // 10
H, n = e13.long_history(fit, fit, 100)                  # newest first
cand = fit["item"][rB].astype(np.int64); y = fit["label"][rB]
Hb, nb = H[ev], n[ev].astype(int)
pos = np.full(len(rB), 0)
for j in range(99, -1, -1):                             # smallest j wins: most recent occurrence
    hit = (Hb[:, j] == cand) & (nb > j)
    pos[hit] = j + 1
print(f"half-B rows {len(rB)}, repeat rows {np.mean(pos > 0):.4f}")
for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 5), (6, 20), (21, 100)]:
    m = (pos >= lo) & (pos <= hi)
    print(f"  last clicked {lo:3d}-{hi:3d} clicks back: {m.sum():7d} rows  click rate {y[m].mean():.4f}  "
          f"mean prediction {pb[m].mean():.4f}  ratio {y[m].mean() / pb[m].mean():.3f}")
rep = pos > 0
print(f"AUC within repeat rows {e13.roc_auc(pb[rep], y[rep]):.4f}; with -position as tie-breaker {e13.roc_auc(pb[rep] - 1e-4 * pos[rep], y[rep]):.4f}")
