# Run from the repository root: .venv/bin/python experiments/lab/refit8.py
# A REFIT OF EVERY ITEM IN THE 8-DIM SUBSPACE. The 32-dim per-item refit failed (lab/item_refit.py): 32 coefficients from
# ~20 clicks. The vectors live in ~8 strong directions (lab/geometry.py), so each item gets 8 coefficients + a bias from its
# own impressions of days 0-5: logit = b_i + beta_i . x, x = the user's unit taste as of the event (mean of the PROJECTED
# vectors of their known clicks, normalised), prior beta_i ~ N(s z_i, 1/lam), z_i = the MF's projected item vector (8-dim
# coordinates) and s a global scale fitted first. Day 6 (MF of days 0-5, one vector seed): AUC of x . beta_i (+ b_i) vs the
# projected cosine, overall and by candidate clicks.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
W = np.load(ROOT / "data/cache/22_vectors_c7.npz")["W_6"].astype(np.float64)
_, U = np.linalg.eigh(W[1:].T @ W[1:]); U8 = U[:, ::-1][:, :8]
Z = W @ U8; Z[0] = 0                                                  # 8-dim coordinates of the projected vectors
H, n = e13.long_history(fit, fit, 100)
def taste(ev):
    h, nn = H[ev], n[ev].astype(int)
    m = (np.arange(100)[None] < nn[:, None])[:, :, None]
    x = (Z[h] * m).sum(1)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
tr_ev = np.flatnonzero(fit["day"] < 6)
X = taste(tr_ev)
it = fit["item"].reshape(-1, 10)[tr_ev].ravel().astype(np.int64)
yl = fit["label"].reshape(-1, 10)[tr_ev].ravel().astype(np.float64)
Xr = np.repeat(X, 10, axis=0)
has = np.repeat(n[tr_ev] > 0, 10)
it, yl, Xr = it[has], yl[has], Xr[has]
Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
# global scale s: logistic regression of the label on x . unit(z_i) with a free intercept (a few Newton steps)
f_ = np.einsum("nd,nd->n", Xr, Zn[it]); a_, s_ = np.log(yl.mean() / (1 - yl.mean())), 0.0
for _ in range(10):
    p = 1 / (1 + np.exp(-(a_ + s_ * f_))); g = np.array([(yl - p).sum(), ((yl - p) * f_).sum()])
    w_ = p * (1 - p); Hm = np.array([[w_.sum(), (w_ * f_).sum()], [(w_ * f_).sum(), (w_ * f_ * f_).sum()]])
    a_, s_ = np.array([a_, s_]) + np.linalg.solve(Hm, g)
print(f"global fit: intercept {a_:.3f}, scale s {s_:.3f}", flush=True)
prior = s_ * Zn
o = np.argsort(it, kind="stable"); it, yl, Xr = it[o], yl[o], Xr[o]
NI = len(Z)
starts = np.searchsorted(it, np.arange(NI + 1))
def refit(lam, iters=10, chunk=500):
    D = 9
    out = np.zeros((NI, D)); out[:, :8] = prior; out[:, 8] = a_
    for c0 in range(1, NI, chunk):
        c1 = min(c0 + chunk, NI); a, b = starts[c0], starts[c1]
        if a == b:
            continue
        Xc = np.c_[Xr[a:b], np.ones(b - a)]; g_id = it[a:b] - c0; ng = c1 - c0
        present, seg = np.unique(g_id, return_index=True)
        mu = np.c_[prior[c0:c1], np.full(ng, a_)]; w = mu.copy()
        reg = np.r_[np.full(8, lam), 1.0]
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.einsum("nd,nd->n", Xc, w[g_id])))
            G = np.zeros((ng, D)); G[present] = np.add.reduceat(Xc * (yl[a:b] - p)[:, None], seg, axis=0)
            G -= reg * (w - mu)
            Hm = np.zeros((ng, D, D))
            Hm[present] = np.add.reduceat((Xc * (p * (1 - p))[:, None])[:, :, None] * Xc[:, None, :], seg, axis=0)
            Hm += np.eye(D) * reg
            step = np.linalg.solve(Hm, G[:, :, None])[:, :, 0]
            nrm = np.linalg.norm(step, axis=1, keepdims=True)
            w += step * np.minimum(1.0, 1.0 / np.maximum(nrm, 1e-12))
        out[c0:c1] = w
    return out
d6 = np.flatnonzero(fit["day"] == 6)
x6 = taste(d6); items6 = fit["item"].reshape(-1, 10)[d6].astype(np.int64); y6 = fit["label"].reshape(-1, 10)[d6].ravel()
k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
cpop = np.bincount(fit["clk_item"][k], minlength=NI)[items6].ravel()
PB = [("0-3", cpop <= 3), ("4-10", (cpop >= 4) & (cpop <= 10)), ("11-30", (cpop >= 11) & (cpop <= 30)), ("31+", cpop >= 31)]
def rep(name, s):
    print(f"{name:30s} AUC {auc(s, y6):.5f} | cand clicks " + "  ".join(f"{g} {auc(s[m], y6[m]):.4f}" for g, m in PB), flush=True)
rep("projected cosine (1 seed)", np.einsum("ed,ekd->ek", x6, Zn[items6]).ravel())
for lam in (1.0, 10.0, 100.0):
    R = refit(lam)
    rep(f"refit lam {lam:g}: x . beta", np.einsum("ed,ekd->ek", x6, R[items6, :8]).ravel())
    rep(f"refit lam {lam:g}: x . beta + b", (np.einsum("ed,ekd->ek", x6, R[items6, :8]) + R[items6, 8]).ravel())
