# Run from the repository root: .venv/bin/python experiments/lab/avg_preds.py OUT.npz A.npz[:key] B.npz[:key] [--w 0.5]
# Rank average of two lab prediction files on the same half-B rows (lab/extra_member.py --save-preds), saved under the key
# "base (= the best member)" so lab/paired_ci.py compares it with either input (2026-10-31, ensemble-diversity screens).
import argparse
import numpy as np
from scipy.stats import rankdata
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--w", type=float, default=0.5)
a = ap.parse_args()
def load(s):
    path, _, key = s.partition(":"); z = np.load(path); return z["rows_B"], z[f"{key or 'base (= the best member)'}|B"]
ra, pa = load(a.a); rb, pb = load(a.b); assert np.array_equal(ra, rb)
np.savez(a.out, rows_B=ra, **{"base (= the best member)|B": (1 - a.w) * rankdata(pa) / len(pa) + a.w * rankdata(pb) / len(pb)})
