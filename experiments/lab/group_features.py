# Run from the repository root (same arguments as lab/extra_member.py, --base-extra = the full set):
#   .venv/bin/python experiments/lab/group_features.py --seq <cache,...> --base-extra name=<cache,...> [...]
# Needs data/cache/22_vectors_c7.npz for the candidate embeddings (MF of days < d, 1 seed).
#
# Intra-impression features on the 26_ensemble_bundle set (which already has, per score, the gap to
# the event's best, rank in the event, softmax share and event log-sum-exp):
#   group z-score and rank percentile within the event of the core scores (MF logit, MF dot,
#   sequence, audience audc / hop2c, co-click sim_sum, item click rate)
#   impression_similarity_variance: variance of the 45 pairwise cosines of the event's candidates
#   dist_to_impression_centroid: 1 - cosine(candidate, mean of the event's 10 candidates)
# Each with binary LightGBM and lambdarank over random groups of 500 events (lab/global_rank.py).
# Half B plain and evaluation-weighted; predictions saved to data/cache/lab_preds_group_features.npz.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
KEY = ["mf_logit", "mf_hist_dot", "seq_logit", "audc_logit", "hop2c_logit", "sim_sum", "item_ctr_logit"]
cols = [base_names.index(k) for k in KEY]
G = base_X[:, cols].astype(np.float64).reshape(-1, 10, len(cols))
zsc = (G - G.mean(1, keepdims=True)) / (G.std(1, keepdims=True) + 1e-6)
rpc = rankdata(G, axis=1) / 10.0
FZ = np.concatenate([zsc.reshape(-1, len(cols)), rpc.reshape(-1, len(cols))], 1).astype(np.float32)
fz_names = [f"{k}_gz" for k in KEY] + [f"{k}_grank" for k in KEY]
# embeddings: MF vectors of the MF trained on the days before each event's day
zv = np.load(ROOT / "data/cache/22_vectors_c7.npz")
evs = rows.reshape(-1, 10)[:, 0] // 10
DISP = np.zeros((len(evs), 1)); DIST = np.zeros((len(evs), 10))
for day in range(2, 7):
    k = fit["day"][evs] == day
    V = zv[f"W_{day}"].astype(np.float64)[fit["item"].reshape(-1, 10)[evs[k]]]
    Vn = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-12)
    Cm = np.einsum("ekd,ejd->ekj", Vn, Vn)
    iu = np.triu_indices(10, 1)
    DISP[k, 0] = Cm[:, iu[0], iu[1]].var(1)
    cen = Vn.mean(1, keepdims=True); cen /= np.linalg.norm(cen, axis=2, keepdims=True) + 1e-12
    DIST[k] = 1 - (Vn * cen).sum(2)
FD = np.concatenate([np.repeat(DISP, 10, 0), DIST.reshape(-1, 1)], 1).astype(np.float32)
fd_names = ["impression_similarity_variance", "dist_to_impression_centroid"]
yB = y[B]
print(f"dist_to_impression_centroid alone (day 6): AUC {auc(-FD[d == 6, 1], y[d == 6]):.5f}", flush=True)

P = {**e13.LGB_PARAMS, "learning_rate": 0.02}
tr_idx = np.flatnonzero(tr); ev_tr = ev[tr_idx]
uev = np.unique(ev_tr); np.random.default_rng(0).shuffle(uev)
rank_of = np.empty(uev.max() + 1, np.int64); rank_of[uev] = np.arange(len(uev))
key = rank_of[ev_tr]; o = np.argsort(key, kind="stable"); idx = tr_idx[o]; sizes = np.bincount(key[o] // 500)
saved = {"rows_B": np.flatnonzero(B)}
def run(label, Xm, nm, rank=False):
    if rank:
        p = {**P, "objective": "lambdarank", "metric": "auc", "lambdarank_truncation_level": 10000, "label_gain": [0, 1]}
        m = lgb.train(p, lgb.Dataset(Xm[idx], y[idx], group=sizes, feature_name=nm), 8000,
                      valid_sets=[lgb.Dataset(Xm[A], y[A], group=[int(A.sum())])], callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        m = lgb.train(P, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                      valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration); saved[f"{label}|B"] = pb
    print(f"  {label:44s} half-B {auc(pb, yB):.5f}  eval-weighted {wauc(pb, yB, W_EVAL[B]):.5f}  ({m.best_iteration} trees)", flush=True)
XA = np.concatenate([base_X, FZ, FD], 1); nA = base_names + fz_names + fd_names
for rank in (False, True):
    tag = "lambdarank G500" if rank else "binary"
    run(f"{tag}: current", base_X, base_names, rank)
    run(f"{tag}: + group z / rank", np.concatenate([base_X, FZ], 1), base_names + fz_names, rank)
    run(f"{tag}: + group z / rank + dispersion", XA, nA, rank)
np.savez(str(CACHE / "lab_preds_group_features.npz"), **saved)
