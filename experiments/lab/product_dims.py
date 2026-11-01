# Run from the repository root: .venv/bin/python experiments/lab/product_dims.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Needs the MF vectors: run first
#   .venv/bin/python experiments/22_product_dims/product_dims.py --cutoff 7 \
#       --save-vectors data/cache/22_vectors_c7.npz
#
# The pipeline gives LightGBM the DOT PRODUCT of the 32-dim user and item vectors -- one number
# out of 32. Here the 32 element-wise products hv * q are added as features (plus, as a second
# variant, the candidate's own 32 dims), so LightGBM can learn which dimensions matter.
# Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B (baseline 0.83571).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
VEC = ROOT / "data" / "cache" / "22_vectors_c7.npz"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
z = np.load(str(VEC))
auc = e13.roc_auc
DIM = z["W_2"].shape[1]
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))

PROD = np.zeros((len(rows), DIM), np.float32)     # hv * q per row
CAND = np.zeros((len(rows), DIM), np.float32)     # q per row
for day in range(2, 7):
    W, hv, evd = z[f"W_{day}"], z[f"hv_{day}"].astype(np.float32), z[f"ev_{day}"]
    items = fit["item"].reshape(-1, 10)[evd]                       # (n_ev, 10)
    r = pos[(evd[:, None] * 10 + np.arange(10)).ravel()]
    q = W[items.ravel()]                                           # (n_ev*10, DIM)
    PROD[r] = q * np.repeat(hv, 10, axis=0)
    CAND[r] = q
    print(f"day {day}: {len(evd):,} events; mean |hv*q| {np.abs(PROD[r]).mean():.4f}", flush=True)

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]
pnames = [f"prod_{i}" for i in range(DIM)]
cnames = [f"cand_{i}" for i in range(DIM)]

# the 1-seed dot product these vectors imply, as a reference against the 15-seed scalar
dot = PROD.sum(1)
print(f"\nday-6 AUC: 1-seed dot product from these vectors {auc(dot[d == 6], y[d == 6]):.5f}, "
      f"15-seed mf_logit {auc(X[d == 6, names.index('mf_logit')], y[d == 6]):.5f}", flush=True)

import lightgbm as lgb
def run(label, Xm, nm):
    m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 3000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    print(f"  {label:38s} half-A {m.best_score['valid_0']['auc']:.5f}  half-B {auc(pb, y[B]):.5f}  "
          f"({m.best_iteration} trees)", flush=True)
    return m
print()
run("13 as is", X, names)
m = run("+ 32 hv*q product dims", np.concatenate([X, PROD], 1), names + pnames)
run("+ product dims + candidate dims", np.concatenate([X, PROD, CAND], 1), names + pnames + cnames)
imp = sorted(zip(m.feature_importance("gain"), names + pnames), reverse=True)[:12]
print("\ntop features by gain (with product dims):")
for g, n in imp:
    print(f"  {n:24s} {g:,.0f}")
