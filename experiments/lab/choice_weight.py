# Run from the repository root: .venv/bin/python experiments/lab/choice_weight.py OUTDIR
# CHOICE WEIGHT (2026-11-01). Alone, the SET40 choice model beats the SET40 LightGBM on day 6 (0.8455 vs 0.8451) and day 4
# (0.8342 vs 0.8337), yet the 42 build gives it a quarter of the rank average (lgb, xgb, choice, shallow). Lab analogue of the
# build (no XGBoost lab predictions; XGBoost ranks like LightGBM): LightGBM : W2 : choice = 2(1-w)/3 : (1-w)/3 : w. Writes one
# predictions file per (day, w) for lab/paired_ci.py; w = 0.25 is the build as it is.
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
K = "base (= the best member)|B"
for T in (6, 5, 4):
    L = np.load(f"data/cache/box/lab_preds_boxbase_d{T}.npz"); W = np.load(f"data/cache/lab_preds_div_W2_d{T}.npz")
    C = np.load(f"data/cache/box/lab_preds_chC0_d{T}.npz")
    assert np.array_equal(L["rows_B"], W["rows_B"]) and np.array_equal(L["rows_B"], C["rows_B"])
    n = len(L["rows_B"]); rl, rw, rc = (rankdata(z[K]) / n for z in (L, W, C))
    for w in (0.25, 0.35, 0.45, 0.6):
        np.savez(out / f"cw{int(w * 100):02d}_d{T}.npz", rows_B=L["rows_B"], **{K: 2 * (1 - w) / 3 * rl + (1 - w) / 3 * rw + w * rc})
print("wrote", sorted(p.name for p in out.glob("cw*")))
