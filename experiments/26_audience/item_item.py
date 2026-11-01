#!/usr/bin/env python3
"""
TechArena CTR -- experiment 26: the audience view at FULL RANK, from raw click counts, written as
extra-group caches for 24_ensemble/ensemble.py (--extra ii1=... etc.).

What: experiment 26's audience A_i is the mean taste of item i's clickers in a 32-dimensional MF
space. Here it is kept as a distribution over items: A[i, j] = mean over i's clicks of the share
of the clicker's OTHER clicks that went to item j (rows sum to ~1). P_j = the same share over all
clicks (the expected A for an unrelated item). For an event with known clicks H and candidate i:
    ii1   mean_{j in H} A[i, j]                                  (raw audience overlap)
    ii2   mean_{j in H} A[i, j] / P_j                            (lift over popularity)
    ii3   mean_{j in H} log((na_i A[i, j] + k P_j) / ((na_i + k) P_j))
          shrunk log-lift, k = 5 pseudo-clicks (naive-Bayes-like; na_i = i's audience clicks)
The candidate itself is left out of A (repeat clicks are covered elsewhere).

Why: the MF compresses "who clicks i" to 32 numbers; item-level detail (this item's clickers
also clicked exactly these items) is lost. The base co-click features pair a click only with the
user's previous 20 clicks and are not normalised by popularity; EASE (ruled out) fits a
regression instead of using the counts.

Everything as of each fit day d (clicks before d); the test events use the clicks before the
cutoff. Columns per cache: [score scaled to std 3, log(1 + na_i), same, score]; days 0-1 NaN.

    .venv/bin/python experiments/26_audience/item_item.py --out-prefix data/cache/26ii
"""
import argparse, importlib.util, time
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
K_SHRINK = 5.0
NAMES = ["ii1", "ii2", "ii3"]


def audience_matrices(fit, day, n_items):
    """A (items x items, csr, diagonal removed), na, P from the clicks before `day`."""
    k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
    cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
    n_users = int(cu.max()) + 1
    n = np.bincount(cu, minlength=n_users).astype(np.float64)
    C = sparse.csr_matrix((np.ones(len(cu)), (cu, ci)), shape=(n_users, n_items))
    w = np.where(n > 1, 1.0 / np.maximum(n - 1, 1), 0.0)
    M = (C.T @ sparse.diags(w) @ C).tocsr()          # M[i, j] = sum_c C_ci C_cj / (n_c - 1)
    M.setdiag(0)
    M.eliminate_zeros()
    na = np.bincount(ci[n[cu] > 1], minlength=n_items).astype(np.float64)
    A = (sparse.diags(1.0 / np.maximum(na, 1)) @ M).tocsr()
    col = np.asarray(M.sum(0)).ravel()
    P = col / col.sum()
    return A, na, np.maximum(P, 1e-12)


def transforms(A, na, P):
    """The three score matrices with A's sparsity pattern, plus ii3's per-item constant."""
    A = A.tocoo()
    r, c, v = A.row, A.col, A.data
    shape = A.shape
    L1 = sparse.csr_matrix((v, (r, c)), shape=shape)
    L2 = sparse.csr_matrix((v / P[c], (r, c)), shape=shape)
    L3 = sparse.csr_matrix((np.log1p(na[r] * v / (K_SHRINK * P[c])), (r, c)), shape=shape)
    return [L1, L2, L3], np.log(K_SHRINK / (na + K_SHRINK))


def score_events(items, hist, hlen, Ls, const3, n_items, chunk=2000):
    """items (E x 10), hist (E x H, 0 = padding), hlen (E,) -> (E*10, 3) scores."""
    E = len(items)
    out = np.zeros((E * 10, len(Ls)), np.float64)
    for s in range(0, E, chunk):
        e = np.arange(s, min(s + chunk, E))
        h, hl = hist[e], hlen[e].astype(np.float64)
        rr, cc = np.nonzero(h)
        Hn = sparse.csr_matrix((1.0 / hl[rr], (rr, h[rr, cc].astype(np.int64))), shape=(len(e), n_items))
        HnT = Hn.T.tocsr()
        cand = items[e].ravel().astype(np.int64)
        uc, inv = np.unique(cand, return_inverse=True)
        ev_local = np.repeat(np.arange(len(e)), 10)
        for j, L in enumerate(Ls):
            R = (L[uc] @ HnT).toarray()                 # unique candidates x events of the chunk
            out[s * 10:(s + len(e)) * 10, j] = R[inv, ev_local]
    out[:, 2] += const3[items.ravel().astype(np.int64)] * (np.repeat(hlen, 10) > 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>_ii1_c7.npz, _ii2_, _ii3_")
    a = ap.parse_args()
    fit, test, _, _, n_items = e13.load_base(a.base)
    hf, lf = e13.long_history(fit, fit, e13.HIST_LONG)
    ht, lt = e13.long_history(test, fit, e13.HIST_LONG)
    n_fit = len(fit["item"])
    out_f = {k: np.full((n_fit, 3), np.nan, np.float32) for k in NAMES}
    for day in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
        t0 = time.time()
        A, na, P = audience_matrices(fit, day, n_items)
        Ls, c3 = transforms(A, na, P)
        evd = np.flatnonzero(fit["day"] == day)
        items = fit["item"].reshape(-1, e13.EV_SIZE)[evd]
        sc = score_events(items, hf[evd], lf[evd], Ls, c3, n_items)
        r = (evd[:, None] * e13.EV_SIZE + np.arange(e13.EV_SIZE)).ravel()
        ln = np.log1p(na[items.ravel()])
        y = fit["label"][r]
        for j, k in enumerate(NAMES):
            out_f[k][r] = np.stack([sc[:, j], ln, ln], 1)
        e13.log(f"day {day}: A nnz {A.nnz / 1e6:.1f}M, " + ", ".join(
            f"{k} AUC {e13.roc_auc(sc[:, j], y):.5f}" for j, k in enumerate(NAMES)) + f" ({time.time() - t0:.0f}s)")
    A, na, P = audience_matrices(fit, e13.OFFICIAL_CUTOFF, n_items)
    Ls, c3 = transforms(A, na, P)
    items = test["item"].reshape(-1, e13.EV_SIZE)
    sc = score_events(items, ht, lt, Ls, c3, n_items)
    ln = np.log1p(na[items.ravel()])
    for j, k in enumerate(NAMES):
        f = out_f[k]
        t = np.stack([sc[:, j], ln, ln], 1).astype(np.float32)
        scale = 3.0 / np.nanstd(f[:, 0])
        f[:, 0] *= scale; t[:, 0] *= scale
        path = f"{a.out_prefix}_{k}_c7.npz"
        np.savez(path, mf_fit=np.concatenate([f, f[:, :1]], 1), mf_test=np.concatenate([t, t[:, :1]], 1))
        e13.log(f"{k}: wrote {path} (test mean {t[:, 0].mean():.3f} sd {t[:, 0].std():.3f}, "
                f"fit mean {np.nanmean(f[:, 0]):.3f} sd {np.nanstd(f[:, 0]):.3f})")


if __name__ == "__main__":
    main()
