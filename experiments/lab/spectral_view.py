# Run from the repository root: .venv/bin/python experiments/lab/spectral_view.py [--k 16] [--out data/cache/47_spec_c7.npz]
# SPECTRAL TASTE VIEW (2026-11-03). The projected cosine (35/40) keeps the 8 strongest principal directions of the item
# vectors with equal weight -- a fixed metric that was the largest representation win. Here the metric is left to the model:
# for every principal direction e_k of a seed's item vectors (eigenvectors of W'W, strongest first) the row gets
# f_k = (u . e_k) x (v . e_k), u = unit mean of the user's known clicks' unit vectors (up to 100), v = the candidate's unit
# vector. sum_k f_k is the plain cosine; the top-8 sum is the projected one. f_k does not change when e_k flips sign, so it is
# comparable across fit days and seeds (each day's MF is trained separately); averaged over the 10 seeds of 40's vectors.
# Columns k = 0..K-1; days 0-1 NaN, empty histories 0. Used as extra choice-model inputs (lab/choice_model.py --add).
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--out", default="data/cache/47_spec_c7.npz")
ap.add_argument("--raw", action="store_true", help="no unit normalisation: u = mean raw vector of the known clicks, v = raw "
                "candidate vector (keeps norms, i.e. the dot product's decomposition instead of the cosine's)")
ap.add_argument("--family", default="40", help="vector files <family>_vectors_c7_s0..9 (40, or 41 = decayed MF)")
ap.add_argument("--var", action="store_true", help="instead: per-direction SPREAD of the user's known clicks, sd_j of "
                "(h_j . e_k) over the unit history vectors h_j (event-level; taste uncertainty per direction)")
ap.add_argument("--sq", action="store_true", help="2026-11-03: instead 2K sign-free LOADINGS per direction: (v . e_k)^2 of "
                "the candidate and (u . e_k)^2 of the user's unit taste (direction-specific item prominence and taste concentration)")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
EV, K = e13.EV_SIZE, a.k
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
files = [ROOT / f"data/cache/{a.family}_vectors_c7_s{s}.npz" for s in range(10)]


def block(H, n, items, W, U):
    Wn = W if a.raw else W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    P = Wn @ U                                                   # unit item vectors in the eigenbasis
    m = (np.arange(H.shape[1])[None] < n[:, None]).astype(np.float32)
    u = (P[H] * m[:, :, None]).sum(1)
    u = u / np.maximum(m.sum(1), 1)[:, None] if a.raw else u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    if a.var:
        c = P[H] * m[:, :, None]; cnt = np.maximum(m.sum(1), 1)[:, None]
        sd = np.sqrt(np.maximum((c ** 2).sum(1) / cnt - (c.sum(1) / cnt) ** 2, 0))
        out = np.repeat(sd[:, None, :], items.shape[1], 1).astype(np.float32)
        out[n == 0] = 0
        return out
    if a.sq:
        out = np.concatenate([P[items] ** 2, np.repeat((u ** 2)[:, None, :], items.shape[1], 1)], 2).astype(np.float32)
        out[n == 0] = 0
        return out
    out = P[items] * u[:, None, :]                               # (events, 10, K)
    out[n == 0] = 0
    return out


def scores(H, n, items, key):
    out = np.zeros(items.shape + (2 * K if a.sq else K,), np.float32)
    for f in files:
        W = np.load(f)[key].astype(np.float32)
        C = W[1:].T.astype(np.float64) @ W[1:].astype(np.float64)
        _, U = np.linalg.eigh(C)
        U = U[:, ::-1][:, :K].astype(np.float32)
        for s in range(0, len(items), 5000):
            sl = slice(s, s + 5000)
            out[sl] += block(H[sl], n[sl].astype(int), items[sl], W, U) / len(files)
    return out.reshape(-1, out.shape[-1])


hf, lf = e13.long_history(fit, fit, e13.HIST_LONG); ht, lt = e13.long_history(test, fit, e13.HIST_LONG)
out_f = np.full((len(fit["item"]), 2 * K if a.sq else K), np.nan, np.float32)
for day in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
    evd = np.flatnonzero(fit["day"] == day)
    r = (evd[:, None] * EV + np.arange(EV)).ravel()
    out_f[r] = scores(hf[evd], lf[evd], fit["item"].reshape(-1, EV)[evd], f"W_{day}")
    s = out_f[r]
    if a.sq:
        e13.log(f"day {day}: loadings written"); continue
    e13.log(f"day {day}: AUC cos {e13.roc_auc(s.sum(1), fit['label'][r]):.5f}  top8 {e13.roc_auc(s[:, :8].sum(1), fit['label'][r]):.5f}  "
            + " ".join(f"{e13.roc_auc(s[:, k], fit['label'][r]):.3f}" for k in range(K)))
t = scores(ht, lt, test["item"].reshape(-1, EV), "W_test")
np.savez(ROOT / a.out, mf_fit=out_f, mf_test=t)
e13.log(f"wrote {a.out}")
