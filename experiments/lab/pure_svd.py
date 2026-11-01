# Run from the repository root: .venv/bin/python experiments/lab/pure_svd.py
# A CONVERGED spectral estimator instead of a trained MF. The audience view (one propagation step of
# clicks through users) and its hops 2-3 each added information, which suggests the fully converged
# version -- a truncated SVD of the (normalised) user x item click matrix, users folded in from their
# known clicks (PureSVD) -- may be a new view. EASE / item-item were full-rank and popularity-dominated;
# here the rank is low and item columns are normalised by popularity^alpha.
# Day 6: matrix from the click log before day 6 (warm-up included); each day-6 event's user row =
# its known clicks (sqrt of counts), score_i = (r_u V) . V_i. Alone and rank corr with MF30.
import importlib.util
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
CUT = e13.TRAIN_START + 6 * e13.DAY
k = fit["clk_ts"] < CUT
cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
nu = int(fit["clk_user"].max()) + 1
C = sp.csr_matrix((np.ones(len(cu)), (cu, ci)), shape=(nu, n_items))
C.data = np.sqrt(C.data)                                       # set-like: repeats count sqrt
pop = np.asarray((C > 0).sum(0)).ravel()

evd = np.flatnonzero(fit["day"] == 6)
items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
H, n = e13.long_history(fit, fit, 100)
H, n = H[evd], n[evd].astype(int)
rr = np.repeat(np.arange(len(evd)), n); cc = H[np.arange(H.shape[1])[None, :] < n[:, None]]
R = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(evd), n_items)); R.data = np.sqrt(R.data)
mf30 = np.mean([np.load(ROOT / "data/cache" / f)["mf_fit"] for f in
                ("13_mf_c7_bb1fcb0a9c89.npz", "15_mf_c7_46f30c940a6a.npz")], 0)
mf6 = mf30[(evd[:, None] * 10 + np.arange(10)).ravel(), 0]
print(f"MF30 alone on day 6: {auc(mf6, y):.5f}", flush=True)
h = np.repeat(n, 10)
groups = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21+", h >= 21)]

rowsum = np.asarray(C.sum(1)).ravel()
for c_s, alpha in ((5, 1.0), (20, 1.0), (20, 0.75), (50, 1.0)):
    colw = 1.0 / (pop + c_s) ** (alpha / 2)
    Cn = sp.diags(1.0 / np.sqrt(np.maximum(rowsum, 1))) @ C @ sp.diags(colw)
    for rank in (32, 64, 128):
        U, s, Vt = svds(Cn.astype(np.float32), k=rank)
        for back in (True, False):
            V = (Vt.T * (colw[:, None] if back else 1.0)).astype(np.float64)
            Rw = R @ sp.diags(colw)                          # user row in the normalised space
            sc = np.einsum("ed,ekd->ek", Rw @ (Vt.T.astype(np.float64)), V[items]).ravel()
            sc_n = sc / np.repeat(np.maximum(np.asarray(R.sum(1)).ravel(), 1), 10)
            rc = np.corrcoef(rankdata(sc_n), rankdata(mf6))[0, 1]
            bl = max(((w, auc(rankdata(mf6) + w * rankdata(sc_n), y)) for w in (0.1, 0.25, 0.5)), key=lambda t: t[1])
            print(f"c {c_s:2d} alpha {alpha} rank {rank:3d} back {int(back)}: alone {auc(sc_n, y):.5f}  " +
                  "  ".join(f"{g} {auc(sc_n[m], y[m]):.4f}" for g, m in groups) +
                  f"  rank corr MF30 {rc:.3f}  best blend w={bl[0]} {bl[1]:.5f}", flush=True)
