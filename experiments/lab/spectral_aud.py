# Run from the repository root: .venv/bin/python experiments/lab/spectral_aud.py [--k 16] [--out data/cache/47_specaud_c7.npz]
# SPECTRAL AUDIENCE VIEW (2026-11-03): the per-direction decomposition (lab/spectral_view.py) of 45's soft-projected cosine
# audience score (soft_aud <- 40c_aud: 26_audience/audience.py --cosine --unit-clickers --soft-proj 20, 40's vectors seeds
# 0-4). Per row f_k = (u . e_k) x (c_i . e_k), u = the user's unit taste (soft-projected unit vectors of the known clicks),
# c_i = unit(A_i - B_i) the candidate's audience minus exposure taste, e_k the principal directions of the item vectors.
# Sign-invariant per direction, so comparable across fit days and seeds; averaged over the 5 seeds.
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--out", default="data/cache/47_specaud_c7.npz"); a = ap.parse_args()
spec = importlib.util.spec_from_file_location("aud", ROOT / "experiments/26_audience/audience.py")
aud = importlib.util.module_from_spec(spec); spec.loader.exec_module(aud)
aud.COSINE, aud.UNIT_CLICKERS, aud.SOFT_PROJ = True, True, 20.0
e13 = aud.e13; EV, K = e13.EV_SIZE, a.k
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
files = [ROOT / f"data/cache/40_vectors_c7_s{s}.npz" for s in range(5)]
unit = lambda X: X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def feats(Wraw, day, H, n, items):
    _, U = np.linalg.eigh(Wraw[1:].T @ Wraw[1:]); U = U[:, ::-1][:, :K]
    W = aud.soft_project(fit, Wraw, day)
    W = unit(W); W[0] = 0.0
    A, _, B = aud.audience(fit, W, day)
    Cp = unit(A - B) @ U; Wp = W @ U
    out = np.zeros(items.shape + (K,), np.float32)
    for s in range(0, len(items), 5000):
        sl = slice(s, s + 5000)
        m = (np.arange(H.shape[1])[None] < n[sl, None])[:, :, None]
        u = unit((Wp[H[sl]] * m).sum(1))
        o = Cp[items[sl]] * u[:, None, :]
        o[n[sl] == 0] = 0
        out[sl] = o
    return out


hf, lf = e13.long_history(fit, fit, e13.HIST_LONG); ht, lt = e13.long_history(test, fit, e13.HIST_LONG)
out_f = np.full((len(fit["item"]), K), np.nan, np.float32)
for day in range(e13.FIRST_MODEL_DAY, e13.OFFICIAL_CUTOFF):
    evd = np.flatnonzero(fit["day"] == day)
    items = fit["item"].reshape(-1, EV)[evd]
    acc = np.mean([feats(np.load(f)[f"W_{day}"].astype(np.float64), day, hf[evd], lf[evd].astype(int), items) for f in files], 0)
    r = (evd[:, None] * EV + np.arange(EV)).ravel()
    out_f[r] = acc.reshape(-1, K)
    s = out_f[r]
    e13.log(f"day {day}: AUC sum {e13.roc_auc(s.sum(1), fit['label'][r]):.5f}  top8 {e13.roc_auc(s[:, :8].sum(1), fit['label'][r]):.5f}")
t = np.mean([feats(np.load(f)["W_test"].astype(np.float64), e13.OFFICIAL_CUTOFF, ht, lt.astype(int), test["item"].reshape(-1, EV))
             for f in files], 0).reshape(-1, K)
np.savez(ROOT / a.out, mf_fit=out_f, mf_test=t.astype(np.float32))
e13.log(f"wrote {a.out}")
