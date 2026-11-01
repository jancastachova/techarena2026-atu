# Run from the repository root (same arguments as lab/extra_member.py, --base-extra = the full set):
#   .venv/bin/python experiments/lab/catboost_rank.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# CatBoost (oblivious trees) on the 26_ensemble_bundle set + the intra-impression group features of
# lab/group_features.py: YetiRank with each event as the ranking group (pairwise inside the
# impression), and Logloss for comparison. Fit days 2-5, early stopping on half A (AUC), half B
# plain and evaluation-weighted; predictions saved to data/cache/lab_preds_catboost.npz.
import argparse, importlib.util, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/group_features.py").read_text()
exec(src[:src.index("P = {**e13.LGB_PARAMS")])                 # base_X, FZ, FD, halves, wauc ...
from catboost import CatBoost, Pool
XA = np.concatenate([base_X, FZ, FD], 1)
saved = {"rows_B": np.flatnonzero(B)}
for loss in ["YetiRank", "Logloss"]:
    t0 = time.time()
    params = dict(loss_function=loss, eval_metric="AUC", iterations=3000, learning_rate=0.08, depth=6,
                  thread_count=8, random_seed=0, od_type="Iter", od_wait=150, verbose=0, use_best_model=True)
    trp = Pool(XA[tr], y[tr], group_id=ev[tr] if loss == "YetiRank" else None)
    vap = Pool(XA[A], y[A], group_id=ev[A] if loss == "YetiRank" else None)
    m = CatBoost(params); m.fit(trp, eval_set=vap)
    pb = m.predict(XA[B]); saved[f"catboost {loss}|B"] = pb
    print(f"  CatBoost {loss:9s} half-B {auc(pb, yB):.5f}  eval-weighted {wauc(pb, yB, W_EVAL[B]):.5f}  "
          f"({m.get_best_iteration()} trees, {time.time() - t0:.0f}s)", flush=True)
np.savez(str(CACHE / "lab_preds_catboost.npz"), **saved)
