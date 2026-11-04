# Run from the repository root: .venv/bin/python experiments/lab/exposure_score.py [--day 6]
# 2026-11-04: does the EXPOSURE geometry predict clicks? Item and user embeddings from the SVD of the normalised user x item
# matrix of shown candidates before day D (re-shows of the user's own clicks removed; no click labels are used for the item
# embedding). On day D, single-score AUC of (a) user exposure embedding . candidate, (b) the user's clicked-item profile in
# exposure space . candidate, by rank; and k-means sharpness of the items in that space vs a Gaussian of the same covariance.
import argparse, importlib.util
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
ROOT = Path.cwd()
ap = argparse.ArgumentParser(); ap.add_argument("--day", type=int, default=6); a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
D = a.day
pre = np.repeat(fit["day"] < D, 10)
u = np.repeat(fit["user"].astype(np.int64), 10)[pre]; it = fit["item"][pre].astype(np.int64); ts = np.repeat(fit["ts"], 10)[pre]
NI = int(max(n_items + 1, it.max() + 1, fit["clk_item"].max() + 1)); NU = int(max(fit["user"].max(), fit["clk_user"].max())) + 1
ck = fit["clk_user"].astype(np.int64) * NI + fit["clk_item"].astype(np.int64)
o = np.lexsort((fit["clk_ts"], ck)); ck_s, ct_s = ck[o], fit["clk_ts"][o]
f1 = np.r_[True, ck_s[1:] != ck_s[:-1]]; pk, pt = ck_s[f1], ct_s[f1]
key = u * NI + it; q = np.minimum(np.searchsorted(pk, key), len(pk) - 1)
keep = ~((pk[q] == key) & (pt[q] < ts)); u, it = u[keep], it[keep]
M = sp.csr_matrix((np.ones(len(u), np.float32), (u, it)), shape=(NU, NI))
du = np.asarray(M.sum(1)).ravel(); di = np.asarray(M.sum(0)).ravel()
Mn = sp.diags(1 / np.sqrt(np.maximum(du, 1))) @ M @ sp.diags(1 / np.sqrt(np.maximum(di, 1)))
R = 32
U, S, Vt = svds(Mn.astype(np.float64), k=R); o = np.argsort(-S); S, U, V = S[o], U[:, o], Vt[o].T
print("singular values:", " ".join(f"{s:.3f}" for s in S[:20]))
evd = np.flatnonzero(fit["day"] == D); items = fit["item"].reshape(-1, 10)[evd].astype(np.int64)
y = fit["label"].reshape(-1, 10)[evd].ravel().astype(float)
H, n = e13.long_history(fit, fit, 100); H, n = H[evd].astype(np.int64), n[evd].astype(int)
msk = (np.arange(100)[None] < n[:, None]).astype(np.float64)[:, :, None]
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
us = fit["user"][evd].astype(np.int64)
for r in (4, 8, 12, 16, 24, 31):
    E = unit(V[:, 1:r + 1] / np.maximum(np.sqrt(di), 1)[:, None] * 1.0); E[di == 0] = 0     # unit item directions
    Ue = unit(U[us, 1:r + 1])                                                               # exposure taste of the user
    Pc = unit((E[H] * msk).sum(1))                                                          # clicked-item profile
    s1 = np.einsum("ed,ekd->ek", Ue, E[items]).ravel(); s2 = np.einsum("ed,ekd->ek", Pc, E[items]).ravel()
    print(f"rank {r:2d}: user exposure . candidate AUC {e13.roc_auc(s1, y):.5f}   clicked profile . candidate AUC {e13.roc_auc(s2, y):.5f}", flush=True)
