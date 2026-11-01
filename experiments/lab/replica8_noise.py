# Run from the repository root: .venv/bin/python experiments/lab/replica8_noise.py
# HOW MUCH ITEM NOISE IS LEFT AFTER THE 8-DIM PROJECTION, on the calibrated 8-dim-truth replica (scale 1.19: MF 0.81736 vs
# real 0.8175). Day 6, MF of days 0-5 (data/cache/replica8_vectors_s{0,1,2}.npz). Every score is the production cosine form:
# x = unit(mean of the history items' vectors), score = x . unit(candidate vector). Item vectors come from
#   true   the truth phi (8-dim)
#   est    the MF vector projected on its own top-8 directions, mapped into the truth space by one least-squares map
#          (click-weighted over items) so that per-item errors can be measured against phi
# and are swapped separately on the history side and on the candidate side, so the loss splits into its two sources.
# Also: per-item cosine(est, true) by candidate clicks, and 3-seed averages (vectors mapped into the same truth space).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/replica8_s1.19.npz"))
T = np.load(ROOT / "data/cache/replica8_s1.19_truth.npz")
auc = e13.roc_auc
PHI = T["phi"].astype(np.float64)
NI = len(PHI)
k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
clicks = np.bincount(fit["clk_item"][k], minlength=NI).astype(np.float64)
_, Up = np.linalg.eigh(PHI[1:].T @ PHI[1:]); Up = Up[:, ::-1][:, :8]
PHI8 = PHI @ Up                                                      # truth in its own 8 coordinates (exact: rank 8)

def mapped(seed):
    W = np.load(ROOT / f"data/cache/replica8_vectors_s{seed}.npz")["W_6"].astype(np.float64)
    _, U = np.linalg.eigh(W[1:].T @ W[1:]); Z = W @ U[:, ::-1][:, :8]
    w = np.sqrt(clicks)[1:, None]
    M, *_ = np.linalg.lstsq(Z[1:] * w, PHI8[1:] * w, rcond=None)
    E = Z @ M; E[0] = 0
    return E

def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)

H, n = e13.long_history(fit, fit, 100)
d6 = np.flatnonzero((fit["day"] == 6) & (n > 0))
h6, n6 = H[d6], n[d6].astype(int)
mask = (np.arange(100)[None] < n6[:, None])[:, :, None]
items6 = fit["item"].reshape(-1, 10)[d6].astype(np.int64)
y6 = fit["label"].reshape(-1, 10)[d6].ravel()
cpop = clicks[items6].ravel()
nb = np.repeat(n6, 10)
PB = [("0-3", cpop <= 3), ("4-10", (cpop >= 4) & (cpop <= 10)), ("11-30", (cpop >= 11) & (cpop <= 30)), ("31+", cpop >= 31)]
KB = [("hist 1-5", nb <= 5), ("6-20", (nb >= 6) & (nb <= 20)), ("21+", nb >= 21)]

def score(Vh, Vc):
    x = unit((Vh[h6] * mask).sum(1))
    return np.einsum("ed,ekd->ek", x, unit(Vc[items6])).ravel()

def rep(name, s):
    print(f"{name:44s} AUC {auc(s, y6):.5f} | " + " ".join(f"{g} {auc(s[m], y6[m]):.4f}" for g, m in PB)
          + " | " + " ".join(f"{g} {auc(s[m], y6[m]):.4f}" for g, m in KB), flush=True)

Es = [mapped(s) for s in range(3)]
Eavg = sum(Es) / 3
print("per-item cosine(est, true) by days 0-5 clicks:")
for name, E in (("seed 0", Es[0]), ("3-seed mean", Eavg)):
    c = np.einsum("nd,nd->n", unit(E), unit(PHI8))
    print(f"  {name:12s} " + "  ".join(f"{lo}-{hi}: {np.mean(c[(clicks >= lo) & (clicks <= hi)]):.3f}"
                                       for lo, hi in ((0, 0), (1, 3), (4, 10), (11, 30), (31, 100), (101, 10**9))), flush=True)
print("items by clicks:", "  ".join(f"{lo}-{hi}: {np.sum((clicks[1:] >= lo) & (clicks[1:] <= hi))}"
                                     for lo, hi in ((0, 0), (1, 3), (4, 10), (11, 30), (31, 100), (101, 10**9))))
rep("true history, true candidates", score(PHI8, PHI8))
rep("est history (seed 0), true candidates", score(Es[0], PHI8))
rep("true history, est candidates (seed 0)", score(PHI8, Es[0]))
rep("est both (seed 0)  ~ production view", score(Es[0], Es[0]))
rep("est both, 3-seed mean vectors", score(Eavg, Eavg))
rep("est both, mean of 3 seed scores", sum(score(E, E) for E in Es) / 3)

# ---- the information limit on the CANDIDATE side in 8 dims: refit every item's 8 coordinates from its own impressions
# of days 0-5 by logistic regression with the TRUE user taste (8 coords) and the true item bias as an offset, prior
# N(m E_i, 1/lam) around the 3-seed MF estimate (m fitted). If this cannot beat the MF candidates, the item side is done.
TH8 = T["theta"].astype(np.float64) @ Up
BI = T["b_item"].astype(np.float64)
tr = np.flatnonzero(fit["day"] < 6)
it = fit["item"].reshape(-1, 10)[tr].ravel().astype(np.int64)
yl = fit["label"].reshape(-1, 10)[tr].ravel().astype(np.float64)
Xr = np.repeat(TH8[fit["user"][tr].astype(np.int64)], 10, axis=0)
off = BI[it]
f_ = np.einsum("nd,nd->n", Xr, Eavg[it]); a_, s_ = -3.0, 1.0
for _ in range(15):                                                 # global intercept + scale on the MF estimate
    p = 1 / (1 + np.exp(-(a_ + off + s_ * f_))); g = np.array([(yl - p).sum(), ((yl - p) * f_).sum()])
    w_ = p * (1 - p); Hm = np.array([[w_.sum(), (w_ * f_).sum()], [(w_ * f_).sum(), (w_ * f_ * f_).sum()]])
    a_, s_ = np.array([a_, s_]) + np.linalg.solve(Hm, g)
print(f"global fit on MF candidates: intercept {a_:.3f}, scale {s_:.3f}", flush=True)
off = off + a_
o = np.argsort(it, kind="stable"); it, yl, Xr, off = it[o], yl[o], Xr[o], off[o]
starts = np.searchsorted(it, np.arange(NI + 1))
def refit(prior, lam, iters=8, chunk=1000):
    out = prior.copy()
    for c0 in range(1, NI, chunk):
        c1 = min(c0 + chunk, NI); a, b = starts[c0], starts[c1]
        if a == b:
            continue
        Xc = Xr[a:b]; g_id = it[a:b] - c0; ng = c1 - c0
        present, seg = np.unique(g_id, return_index=True)
        mu = prior[c0:c1]; w = mu.copy()
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(off[a:b] + np.einsum("nd,nd->n", Xc, w[g_id]))))
            G = np.zeros((ng, 8)); G[present] = np.add.reduceat(Xc * (yl[a:b] - p)[:, None], seg, axis=0)
            G -= lam * (w - mu)
            Hm = np.zeros((ng, 8, 8))
            Hm[present] = np.add.reduceat((Xc * (p * (1 - p))[:, None])[:, :, None] * Xc[:, None, :], seg, axis=0)
            Hm += np.eye(8) * lam
            step = np.linalg.solve(Hm, G[:, :, None])[:, :, 0]
            nrm = np.linalg.norm(step, axis=1, keepdims=True)
            w += step * np.minimum(1.0, 1.0 / np.maximum(nrm, 1e-12))
        out[c0:c1] = w
    return out
for lam in (0.3, 3.0, 30.0):
    R = refit(s_ * Eavg, lam)
    c = np.einsum("nd,nd->n", unit(R), unit(PHI8))
    print(f"lam {lam:g}: per-item cos " + "  ".join(f"{lo}-{hi}: {np.mean(c[(clicks >= lo) & (clicks <= hi)]):.3f}"
          for lo, hi in ((1, 3), (4, 10), (11, 30), (31, 100))), flush=True)
    rep(f"true history, refit candidates lam {lam:g}", score(PHI8, R))
    rep(f"est history, refit candidates lam {lam:g}", score(Eavg, R))
