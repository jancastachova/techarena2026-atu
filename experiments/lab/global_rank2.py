# Run from the repository root (same arguments as lab/extra_member.py, --base-extra = the full set):
#   .venv/bin/python experiments/lab/global_rank2.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# Follow-up of lab/global_rank.py: lambdarank over random groups of G events beat binary at G = 500
# (half B 0.84137 vs 0.84118, eval-weighted 0.82467 vs 0.82445, one run). G 1000 (the largest LightGBM
# allows: 10,000 rows per query), predictions
# saved (data/cache/lab_preds_global_rank.npz) for lab/paired_ci.py, and rank blends with binary.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
P = {**e13.LGB_PARAMS, "learning_rate": 0.02}
saved = {"rows_B": np.flatnonzero(B)}
def show(label, pb):
    saved[f"{label}|B"] = pb
    fb = FIRST[B]
    print(f"  {label:30s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  "
          f"first {wauc(pb[fb], y[B][fb], W_FIRST[B][fb]):.5f}", flush=True)
m = lgb.train(P, lgb.Dataset(base_X[tr], y[tr], feature_name=base_names), 8000,
              valid_sets=[lgb.Dataset(base_X[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
show("binary", m.predict(base_X[B], num_iteration=m.best_iteration))
tr_idx = np.flatnonzero(tr); ev_tr = ev[tr_idx]
uev = np.unique(ev_tr); np.random.default_rng(0).shuffle(uev)
rank_of = np.empty(uev.max() + 1, np.int64); rank_of[uev] = np.arange(len(uev))
key = rank_of[ev_tr]; o = np.argsort(key, kind="stable"); idx = tr_idx[o]
import os
GROUPS = [int(g_) for g_ in os.environ.get("GR_GROUPS", "500,1000").split(",")]
OUT = str(CACHE / os.environ.get("GR_OUT", "lab_preds_global_rank.npz"))
for G in GROUPS:                          # LightGBM caps a query at 10,000 rows (1,000 events)
    sizes = np.bincount(key[o] // G)
    p = {**P, "objective": "lambdarank", "metric": "auc", "lambdarank_truncation_level": 10000, "label_gain": [0, 1]}
    mm = lgb.train(p, lgb.Dataset(base_X[idx], y[idx], group=sizes, feature_name=base_names), 8000,
                   valid_sets=[lgb.Dataset(base_X[A], y[A], group=[int(A.sum())])],
                   callbacks=[lgb.early_stopping(150, verbose=False)])
    show(f"lambdarank G{G}", mm.predict(base_X[B], num_iteration=mm.best_iteration))
    np.savez(OUT, **saved)
r = lambda q: rankdata(q) / len(q)
for G in GROUPS:
    show(f"binary + lambdarank G{G} (rank avg)", r(saved["binary|B"]) + r(saved[f"lambdarank G{G}|B"]))
np.savez(OUT, **saved)
