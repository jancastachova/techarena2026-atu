# Run from the repository root: TEST_DAY=3 .venv/bin/python experiments/lab/ens_compose.py T
# FINAL SHORTLIST ON AN UNUSED DAY (2026-11-01). Day 3 was never a test day (fit on day 2 only, so it is the weakest split).
# Whole ensembles as the builds compose them, rank averages of the lab members trained on that split:
#   A  40 config   lgb, xgb, choice                equal
#   B  42 config   lgb, xgb, choice, shallow (W2)  equal
#   C  B with the choice member at 0.40, the other three 0.20 each
# Files data/cache/lab_preds_d{T}_{lgb,xgb,w2,choice}.npz -> data/cache/lab_preds_d{T}_ens{A,B,C}.npz for lab/paired_ci.py.
import sys
import numpy as np
from scipy.stats import rankdata
T = sys.argv[1]; K = "base (= the best member)|B"
Z = {m: np.load(f"data/cache/lab_preds_d{T}_{m}.npz") for m in ("lgb", "xgb", "w2", "choice")}
r = Z["lgb"]["rows_B"]; assert all(np.array_equal(z["rows_B"], r) for z in Z.values())
R = {m: rankdata(z[K]) / len(r) for m, z in Z.items()}
W = {"A": {"lgb": 1, "xgb": 1, "choice": 1}, "B": {"lgb": 1, "xgb": 1, "choice": 1, "w2": 1},
     "C": {"lgb": .2, "xgb": .2, "choice": .4, "w2": .2}}
for k, w in W.items():
    np.savez(f"data/cache/lab_preds_d{T}_ens{k}.npz", rows_B=r, **{K: sum(v * R[m] for m, v in w.items()) / sum(w.values())})
print("rank corr lgb/xgb/choice/w2:", np.round(np.corrcoef([R[m] for m in ("lgb", "xgb", "choice", "w2")]), 4).tolist())
