# Lean LightGBM + category columns built from the prefix of test day T, as a stacker input (2026-11-04). Fit days 2..T-1, half A of
# day T for early stopping, predictions on half B of day T (the same rows as the other lab members).
#   python experiments/49_bayes_choice/cat_member.py <T> <out.npz> <cat.npz> [<cat.npz> ...]
import sys, os, time
import numpy as np
import lightgbm as lgb
T, out, cats = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
Z = np.load("data/cache/lab_export_lean_d6.npz")
X, y, d = Z["X"], Z["y"].astype(np.float64), Z["d"]
names = [n.replace(" ", "_") for n in Z["names"]]
for c in cats:
    E = np.load(c); assert len(E["X"]) == len(X)
    X = np.concatenate([X, E["X"]], 1); names += [str(n) for n in E["names"]]
lab = "data/cache/box/lab/" + ("lab_preds_d3_lgb.npz" if T == 3 else f"lab_preds_boxbase_d{T}.npz")
rb = np.load(lab)["rows_B"]
B = np.zeros(len(y), bool); B[rb] = True
assert (d[B] == T).all()
A = (d == T) & ~B; tr = d < T
t0 = time.time()
P = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, deterministic=True, force_row_wise=True, num_threads=int(os.environ.get("NTHREADS", 9)))
dtr = lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=True)
m = lgb.train(P, dtr, 8000, valid_sets=[lgb.Dataset(X[A], y[A], reference=dtr)], callbacks=[lgb.early_stopping(150, verbose=False)])
p = m.predict(X[B], num_iteration=m.best_iteration)
np.savez(out, rows_B=rb, **{"base (= the best member)|B": p})
print(f"T={T}: {tr.sum():,} train rows, {m.best_iteration} trees, half-A AUC {m.best_score['valid_0']['auc']:.5f}, {time.time() - t0:.0f}s", flush=True)
