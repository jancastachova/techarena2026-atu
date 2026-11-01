# Run from the repository root (same arguments as lab/extra_member.py, --base-extra = the full set):
#   .venv/bin/python experiments/lab/global_rank.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# The metric is GLOBAL AUC (every row against every row), LightGBM optimises row log loss. Ranking
# objectives with one EVENT per group (RESULTS.md: lambdarank 0.7808, rank_xendcg 0.7802 vs 0.7863)
# optimise only the within-event order. Here the groups are many events at once (random, of G events),
# so the ranking objective sees cross-event pairs, i.e. a direct surrogate of global AUC. Bundle set,
# lr 0.02, early stopping on half A (AUC), half B plain and evaluation-weighted.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
P = {**e13.LGB_PARAMS, "learning_rate": 0.02}
def show(label, m, Xm):
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    print(f"  {label:34s} half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}  ({m.best_iteration} trees)", flush=True)
m = lgb.train(P, lgb.Dataset(base_X[tr], y[tr], feature_name=base_names), 8000,
              valid_sets=[lgb.Dataset(base_X[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
show("binary (current)", m, base_X)
tr_idx = np.flatnonzero(tr)
ev_tr = ev[tr_idx]
for G in [50, 500]:
    for obj in ["rank_xendcg", "lambdarank"]:
        # random groups of G events: shuffle events, keep the rows of an event together
        uev = np.unique(ev_tr); rng = np.random.default_rng(0); rng.shuffle(uev)
        order_ev = {e: i for i, e in enumerate(uev)}
        key = np.array([order_ev[e] for e in ev_tr])
        o = np.argsort(key, kind="stable"); idx = tr_idx[o]
        grp_of_row = key[o] // G
        sizes = np.bincount(grp_of_row)
        p = {**P, "objective": obj, "metric": "auc"}
        if obj == "lambdarank":
            p["lambdarank_truncation_level"] = 10000; p["label_gain"] = [0, 1]
        try:
            mm = lgb.train(p, lgb.Dataset(base_X[idx], y[idx], group=sizes, feature_name=base_names), 8000,
                           valid_sets=[lgb.Dataset(base_X[A], y[A], group=[int(A.sum())])],
                           callbacks=[lgb.early_stopping(150, verbose=False)])
            show(f"{obj}, groups of {G} events", mm, base_X)
        except Exception as e_:
            print(f"  {obj}, groups of {G}: failed ({e_})", flush=True)
