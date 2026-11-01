# Run from the repository root: .venv/bin/python experiments/lab/diag_time_by_history.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Where does experiment 13 lose AUC on first events? Validation cutoff 5 (test = each user's
# first event at/after day 5), model = the same LightGBM procedure as the submission.
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c5_a2ecdd59dd03.npz"))
z = np.load(str(ROOT / "data/cache/13_mf_c5_2889ef80775f.npz"))
model = e13.lgbm_procedure(fit, z["mf_fit"], 5)
rows, X, _ = e13.feature_matrix(test, z["mf_test"], np.arange(len(test["item"])))
p, y = model.predict(X), test["label"].astype(np.int64)
auc = e13.roc_auc
E = len(test["user"])
P, Y = p.reshape(E, 10), y.reshape(E, 10)
print(f"\nfirst events: {E:,}, rows {len(y):,}, click rate {y.mean():.4f}, events with click {Y.any(1).mean():.3f}")
n_clk = np.bincount(fit["clk_user"], minlength=int(test["user"].max()) + 1)
uc = np.repeat(n_clk[test["user"]], 10)
hrs = np.repeat((test["ts"] - (e13.TRAIN_START + 5 * e13.DAY)) / 3600, 10)
print(f"\n{'known clicks':14s} " + " ".join(f"{s:>16s}" for s in ["0-6h", "6-24h", "24-48h"]))
for name, um in [("1-5", (uc >= 1) & (uc <= 5)), ("6-20", (uc >= 6) & (uc <= 20)), ("21-100", (uc > 20) & (uc <= 100))]:
    cells = []
    for a, b in [(0, 6), (6, 24), (24, 48)]:
        m = um & (hrs >= a) & (hrs < b)
        cells.append(f"{auc(p[m], y[m]):.4f} ({m.sum() // 10:>5d})")
    print(f"{name:14s} " + " ".join(f"{c:>16s}" for c in cells))
