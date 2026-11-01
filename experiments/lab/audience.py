# Run from the repository root: .venv/bin/python experiments/lab/audience.py [--days 6]
# Needs data/cache/22_vectors_c7.npz (item vectors W_<d> of an MF trained on days < d, 1 seed).
#
# User-based view: a candidate is represented by its AUDIENCE -- the mean taste of the users who
# clicked it (each clicker's taste = mean vector of their OTHER known clicks), instead of its own
# learned vector. For a rare item (3-9 clicks; RESULTS: underfit for lack of information) the
# audience pools the clickers' whole histories (~26 clicks each). Everything as of day d: clicks
# before day d, vectors of an MF trained on days < d. Variants:
#   aud      u . mean_{clickers}(taste)
#   aud_c    u . (mean_{clickers}(taste) - mean_{shown-to}(taste))   (contrast with who sees it)
#   aud_s    shrunk toward the item's own vector: (k * w + n * aud) / (k + n)
# Screening: day-6 AUC of each score alone vs the MF dot, and the rank correlation with it.
import argparse, importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
ap = argparse.ArgumentParser()
ap.add_argument("--days", default="6")
ap.add_argument("--more", action="store_true", help="contrast weights and two-hop user vectors")
a = ap.parse_args()
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(str(ROOT / "data/cache/22_vectors_c7.npz"))
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
auc = e13.roc_auc


def audience(W, day):
    """Per item: mean taste of its clickers (clicks before `day`), and of the users it was shown to."""
    k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
    cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
    n_users = int(max(cu.max(), fit["user"].max())) + 1
    S = np.zeros((n_users, W.shape[1])); np.add.at(S, cu, W[ci])
    n = np.bincount(cu, minlength=n_users).astype(np.float64)
    other = (S[cu] - W[ci]) / np.maximum(n[cu] - 1, 1)[:, None]          # clicker's other clicks
    ok = n[cu] > 1
    A = np.zeros_like(W); np.add.at(A, ci[ok], other[ok])
    na = np.bincount(ci[ok], minlength=W.shape[0]).astype(np.float64)
    A /= np.maximum(na, 1)[:, None]
    # users each item was shown to (impressions before day), taste = mean of all their clicks
    ev = np.flatnonzero(fit["day"] < day)
    su = np.repeat(fit["user"][ev].astype(np.int64), 10); si = fit["item"].reshape(-1, 10)[ev].ravel()
    taste = S / np.maximum(n, 1)[:, None]
    Bm = np.zeros_like(W); np.add.at(Bm, si, taste[su])
    nb = np.bincount(si, minlength=W.shape[0]).astype(np.float64)
    Bm /= np.maximum(nb, 1)[:, None]
    return A, na, Bm


for day in [int(x) for x in a.days.split(",")]:
    W = z[f"W_{day}"].astype(np.float64); evd = z[f"ev_{day}"]
    A, na, Bm = audience(W, day)
    hv = z[f"hv_{day}"].astype(np.float64)
    items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
    dot = lambda V: np.einsum("ed,ekd->ek", hv, V[items]).ravel()
    s_mf = dot(W); r_mf = rankdata(s_mf)
    print(f"day {day}: {len(evd):,} events; MF dot AUC {auc(s_mf, y):.5f}; "
          f"candidates with >=1 audience click {np.mean(na[items] > 0):.3f}", flush=True)
    cand = {"aud": A, "aud_c": A - Bm}
    if a.more:
        for lam in [0.5, 1.5, 2.0]:
            cand[f"aud_c lam={lam}"] = A - lam * Bm
        cand["W - Bm"] = W - Bm
        cand["W + aud_c"] = W / W.std() + (A - Bm) / (A - Bm).std()
    else:
        for kk in [3, 10, 30]:
            cand[f"aud_s k={kk}"] = (kk * W + na[:, None] * A) / (kk + na[:, None])
    nclk = np.bincount(fit["clk_item"][fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY], minlength=W.shape[0])[items].ravel()
    rare = (nclk >= 1) & (nclk <= 9)
    for name, V in cand.items():
        s = dot(V)
        print(f"  {name:12s} AUC {auc(s, y):.5f}  rare(1-9 clicks) {auc(s[rare], y[rare]):.5f} vs MF {auc(s_mf[rare], y[rare]):.5f}  "
              f"rank corr with MF {np.corrcoef(rankdata(s), r_mf)[0, 1]:.4f}", flush=True)
    if a.more:
        # two hops on the user side: the user's taste = mean AUDIENCE of their known clicks
        H, n = hist[evd], hlen[evd]
        msk = (np.arange(H.shape[1])[None, :] < n[:, None])[:, :, None]
        for uname, U in [("u=mean aud_c", A - Bm), ("u=mean aud", A)]:
            u2 = (U[H] * msk).sum(1) / np.maximum(n, 1)[:, None]
            for cname, V in [("x W", W), ("x aud_c", A - Bm)]:
                s = np.einsum("ed,ekd->ek", u2, V[items]).ravel()
                print(f"  {uname} {cname:8s} AUC {auc(s, y):.5f}  rare {auc(s[rare], y[rare]):.5f}  "
                      f"rank corr with MF {np.corrcoef(rankdata(s), r_mf)[0, 1]:.4f}", flush=True)
