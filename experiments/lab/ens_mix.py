# Run from the repository root: .venv/bin/python experiments/lab/ens_mix.py OUT.npz lgb=P.npz:0.2 xgb=P.npz:0.2 ...
# The build's ensemble on a lab split (2026-11-03): weighted rank average of member half-B predictions (lab/paired_ci.py
# format), e.g. 45's composition lgb 0.2 / xgb 0.2 / shallow 0.2 / choice 0.4 with any choice variant swapped in.
import sys
import numpy as np
from scipy.stats import rankdata
K = "base (= the best member)|B"
out, parts = sys.argv[1], [s.split("=", 1)[1].rsplit(":", 1) for s in sys.argv[2:]]
Z = [(np.load(p), float(w)) for p, w in parts]
r = Z[0][0]["rows_B"]; assert all(np.array_equal(z["rows_B"], r) for z, _ in Z)
np.savez(out, rows_B=r, **{K: sum(w * rankdata(z[K]) / len(r) for z, w in Z) / sum(w for _, w in Z)})
print("wrote", out)
