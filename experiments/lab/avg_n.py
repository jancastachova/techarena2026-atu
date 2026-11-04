# Run from the repository root: .venv/bin/python experiments/lab/avg_n.py OUT.npz A.npz B.npz [C.npz ...]
# Equal rank average of any number of lab prediction files on the same half-B rows (key "base (= the best member)"),
# saved under that key for lab/paired_ci.py (2026-11-01; lab/avg_preds.py does two with a weight).
import sys
import numpy as np
from scipy.stats import rankdata
out, files = sys.argv[1], sys.argv[2:]
Z = [np.load(f) for f in files]
r = Z[0]["rows_B"]
assert all(np.array_equal(z["rows_B"], r) for z in Z)
p = sum(rankdata(z["base (= the best member)|B"]) for z in Z) / (len(Z) * len(r))
np.savez(out, rows_B=r, **{"base (= the best member)|B": p})
