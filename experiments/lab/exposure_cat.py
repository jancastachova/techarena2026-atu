# Run from the repository root: .venv/bin/python experiments/lab/exposure_cat.py [--day 7] [--k 8]
# 2026-11-04: are there DISCRETE ITEM CATEGORIES in the candidate generator? An outside plan says slates are drawn as
#   q(category | user) ~ global_category_share * exp(coupling * user_category_affinity).
# We only ever looked for clusters in click-trained MF vectors (lab/simplex_test.py), never in EXPOSURE. Here: the user x item
# matrix of shown candidates before day D (re-shows of the user's own clicks removed, so only the ordinary generator is left),
# normalised D_u^-1/2 M D_i^-1/2. Printed: singular values (a block structure of k categories gives k leading values, then
# a noise floor), k-means on the unit item embeddings, how sharp the clusters are, and whether a user's exposure histogram
# over the clusters predicts what they are shown and what they click.
import argparse, importlib.util
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
ROOT = Path.cwd()
ap = argparse.ArgumentParser(); ap.add_argument("--day", type=int, default=7); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--rank", type=int, default=24); a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
D = a.day
pre_ev = fit["day"] < D
u = np.repeat(fit["user"].astype(np.int64), 10)[np.repeat(pre_ev, 10)]
it = fit["item"][np.repeat(pre_ev, 10)].astype(np.int64)
ts = np.repeat(fit["ts"], 10)[np.repeat(pre_ev, 10)]
NI = int(max(n_items + 1, it.max() + 1, fit["clk_item"].max() + 1)); NU = int(max(fit["user"].max(), fit["clk_user"].max())) + 1
ck = fit["clk_user"].astype(np.int64) * NI + fit["clk_item"].astype(np.int64)
o = np.lexsort((fit["clk_ts"], ck)); ck_s, ct_s = ck[o], fit["clk_ts"][o]
first = np.r_[True, ck_s[1:] != ck_s[:-1]]; pk, pt = ck_s[first], ct_s[first]        # first click time of each (user, item)
key = u * NI + it
q = np.minimum(np.searchsorted(pk, key), len(pk) - 1)
reshow = (pk[q] == key) & (pt[q] < ts)
print(f"impressions before day {D}: {len(u):,}; re-shows of the user's own clicks removed: {reshow.mean():.4f}")
u, it = u[~reshow], it[~reshow]
M = sp.csr_matrix((np.ones(len(u), np.float32), (u, it)), shape=(NU, NI))
du = np.asarray(M.sum(1)).ravel(); di = np.asarray(M.sum(0)).ravel()
Mn = sp.diags(1 / np.sqrt(np.maximum(du, 1))) @ M @ sp.diags(1 / np.sqrt(np.maximum(di, 1)))
U, S, Vt = svds(Mn.astype(np.float64), k=a.rank)
o = np.argsort(-S); S, U, Vt = S[o], U[:, o], Vt[o]
print("singular values:", " ".join(f"{s:.4f}" for s in S))
# null: the same degrees, candidates shuffled across users
rng = np.random.default_rng(0)
Mr = sp.csr_matrix((np.ones(len(u), np.float32), (u, rng.permutation(it))), shape=(NU, NI))
Mrn = sp.diags(1 / np.sqrt(np.maximum(du, 1))) @ Mr @ sp.diags(1 / np.sqrt(np.maximum(di, 1)))
Sr = np.sort(svds(Mrn.astype(np.float64), k=6, return_singular_vectors=False))[::-1]
print("shuffled    :", " ".join(f"{s:.4f}" for s in Sr))
np.savez("/private/tmp/claude-501/-Users-ivanbibat-Documents-GitHub-techarena2026-atu/bab4a5eb-6bb2-4aad-9f1d-bf2bfb4aa874/scratchpad/expo_svd.npz",
         S=S, U=U.astype(np.float32), V=Vt.T.astype(np.float32), di=di, du=du)
