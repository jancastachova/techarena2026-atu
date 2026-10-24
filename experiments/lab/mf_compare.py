# Run from the repository root: .venv/bin/python experiments/lab/mf_compare.py <mfA.npz> <mfB.npz>
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Compares two MF caches at the LightGBM stage: each alone, both feature sets together, and the
# rank blend of the two models. Fit days 2-5; day 6 split by event: half A for early stopping and
# the blend weight, AUC reported on half B (13's MF gives 0.83571).
import importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
A_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/cache/13_mf_c7_bb1fcb0a9c89.npz"
B_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/cache/20_mf_c7_c6b710c438b5.npz"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mfA, mfB = np.load(A_PATH)["mf_fit"], np.load(B_PATH)["mf_fit"]
auc = e13.roc_auc
row_day = np.repeat(fit["day"], 10)
keep = np.flatnonzero(row_day >= 2)
rows, XA, names = e13.feature_matrix(fit, mfA, keep)
_, XB, _ = e13.feature_matrix(fit, mfB, keep)
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]
n_base = len(e13.BASE_FEATURES)
mf_names = e13.MF_FEATURES + e13.REL_FEATURES[:4]
both_names = names + [f"{n}_b" for n in mf_names]
Xboth = np.concatenate([XA, XB[:, n_base:n_base + len(mf_names)]], 1)

print(f"A = {Path(A_PATH).name}\nB = {Path(B_PATH).name}")
print(f"MF alone on day 6: A {auc(mfA[rows][d == 6, 0], y[d == 6]):.5f}  "
      f"B {auc(mfB[rows][d == 6, 0], y[d == 6]):.5f}", flush=True)

import lightgbm as lgb
def run(label, X, nm):
    m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=nm), 3000,
                  valid_sets=[lgb.Dataset(X[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pa, pb = m.predict(X[A], num_iteration=m.best_iteration), m.predict(X[B], num_iteration=m.best_iteration)
    print(f"  {label:26s} half-A {auc(pa, y[A]):.5f}  half-B {auc(pb, y[B]):.5f}  "
          f"({m.best_iteration} trees)", flush=True)
    return pa, pb

pA = run("LightGBM with MF A", XA, names)
pB = run("LightGBM with MF B", XB, names)
run("LightGBM with both MFs", Xboth, both_names)
r = lambda p: rankdata(p) / len(p)
print(f"  rank correlation of A and B predictions on half B: {np.corrcoef(r(pA[1]), r(pB[1]))[0, 1]:.4f}")
ws = np.linspace(0, 1, 11)
w = ws[int(np.argmax([auc((1 - w) * r(pA[0]) + w * r(pB[0]), y[A]) for w in ws]))]
blend = auc((1 - w) * r(pA[1]) + w * r(pB[1]), y[B])
print(f"  rank blend (w_B {w:.1f} from half A)  half-B {blend:.5f}  "
      f"vs A alone {auc(pA[1], y[B]):.5f}  gain {blend - auc(pA[1], y[B]):+.5f}")
