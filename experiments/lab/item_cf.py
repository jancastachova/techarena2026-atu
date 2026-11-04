# Run from the repository root: .venv/bin/python experiments/lab/item_cf.py --out-prefix data/cache/48cf
# 2026-11-03 (external review: "item-item CF with cosine / BM25 / shrunk similarity"). 26_audience/item_item.py tried raw
# overlap, lift and shrunk log-lift (alone 0.56-0.62, nothing in LightGBM); the normalisations untried there are:
#   cos    co[i, j] / sqrt(n_i n_j)                   (binary user x item click matrix)
#   coss   co[i, j] / (sqrt(n_i n_j) + 20)            (shrunk toward 0 for sparse pairs)
#   bm25   cosine of BM25-weighted user x item rows (k1 1.2, b 0.75 over users' click counts, idf per item)
# Score = mean over the user's known clicks (up to 100) of sim(candidate, click); time-forward like item_item.py (clicks
# before each fit day; the test uses the clicks before the cutoff). Caches as item_item.py: [score std 3, log1p clicks, x2].
import argparse, importlib.util, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("ii", ROOT / "experiments/26_audience/item_item.py")
ii = importlib.util.module_from_spec(spec); spec.loader.exec_module(ii)
e13 = ii.e13
NAMES = ["cos", "coss", "bm25"]


def sims(fit, day, n_items):
    k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
    cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
    n_users = int(cu.max()) + 1
    C = sparse.csr_matrix((np.ones(len(cu)), (cu, ci)), shape=(n_users, n_items))
    C.data[:] = 1.0                                     # binary (duplicates summed by csr -> reset)
    ni = np.asarray(C.sum(0)).ravel()
    co = (C.T @ C).tocsr(); co.setdiag(0); co.eliminate_zeros()
    co = co.tocoo(); r, c, v = co.row, co.col, co.data
    den = np.sqrt(ni[r] * ni[c])
    cos = sparse.csr_matrix((v / den, (r, c)), shape=co.shape)
    coss = sparse.csr_matrix((v / (den + 20.0), (r, c)), shape=co.shape)
    nu = np.asarray(C.sum(1)).ravel(); avg = nu[nu > 0].mean()
    idf = np.log(1 + (n_users - ni + 0.5) / (ni + 0.5))
    Cc = C.tocoo()
    tf = Cc.data * 2.2 / (Cc.data + 1.2 * (0.25 + 0.75 * nu[Cc.row] / avg))
    Bm = sparse.csr_matrix((tf * idf[Cc.col], (Cc.row, Cc.col)), shape=C.shape)
    nb = np.sqrt(np.asarray(Bm.multiply(Bm).sum(0)).ravel()); nb[nb == 0] = 1
    Bn = Bm @ sparse.diags(1 / nb)
    bm = (Bn.T @ Bn).tocsr(); bm.setdiag(0); bm.eliminate_zeros()
    return [cos, coss, bm], ni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    fit, test, _, _, n_items = e13.load_base(a.base)
    hf, lf = e13.long_history(fit, fit, e13.HIST_LONG)
    ht, lt = e13.long_history(test, fit, e13.HIST_LONG)
    out_f = {k: np.full((len(fit["item"]), 3), np.nan, np.float32) for k in NAMES}
    zero = np.zeros(n_items)
    for day in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
        t0 = time.time()
        Ls, ni = sims(fit, day, n_items)
        evd = np.flatnonzero(fit["day"] == day)
        items = fit["item"].reshape(-1, e13.EV_SIZE)[evd]
        sc = ii.score_events(items, hf[evd], lf[evd], Ls, zero, n_items)
        r = (evd[:, None] * e13.EV_SIZE + np.arange(e13.EV_SIZE)).ravel()
        ln = np.log1p(ni[items.ravel()]); y = fit["label"][r]
        for j, k in enumerate(NAMES):
            out_f[k][r] = np.stack([sc[:, j], ln, ln], 1)
        e13.log(f"day {day}: " + ", ".join(f"{k} AUC {e13.roc_auc(sc[:, j], y):.5f}" for j, k in enumerate(NAMES))
                + f" ({time.time() - t0:.0f}s)")
    Ls, ni = sims(fit, e13.OFFICIAL_CUTOFF, n_items)
    items = test["item"].reshape(-1, e13.EV_SIZE)
    sc = ii.score_events(items, ht, lt, Ls, zero, n_items)
    ln = np.log1p(ni[items.ravel()])
    for j, k in enumerate(NAMES):
        f = out_f[k]; t = np.stack([sc[:, j], ln, ln], 1).astype(np.float32)
        s = 3.0 / np.nanstd(f[:, 0]); f[:, 0] *= s; t[:, 0] *= s
        path = f"{a.out_prefix}_{k}_c7.npz"
        np.savez(path, mf_fit=np.concatenate([f, f[:, :1]], 1), mf_test=np.concatenate([t, t[:, :1]], 1))
        e13.log(f"{k}: wrote {path}")


if __name__ == "__main__":
    main()
