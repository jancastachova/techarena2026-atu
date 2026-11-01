# Run from the repository root: .venv/bin/python experiments/lab/user_clusters.py
# ARE USER TASTES DRAWN FROM A FEW ARCHETYPES? lab/simplex_test.py found the ITEM cloud a smooth Gaussian, which leaves the
# user side open: a generator can draw each user's taste from K centres (+ small noise) while items stay isotropic. If so,
# a light user's 1-5 clicks could be snapped to the nearest archetype (a far better prior than the noisy mean).
# Heavy users (>= 50 clicks before day 6), taste = mean of the unit 8-dim projected MF vectors (W_6 of 22, days 0-5) of
# their clicks. Tests, real vs a Gaussian with the same covariance:
#   1. k-means mean cosine to the nearest centre, k = 4 / 8 / 16 / 32
#   2. split-half stability: each user's clicks split in two random halves, both halves assigned to the real k-means
#      centres; agreement rate vs the same with a Gaussian-matched surrogate (halves = truth + independent noise of the
#      size measured from the real halves)
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
rng = np.random.default_rng(0)
W = np.load(ROOT / "data/cache/22_vectors_c7.npz")["W_6"].astype(np.float64)
def unit(A): return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
_, U = np.linalg.eigh(W[1:].T @ W[1:]); Z = unit(W @ U[:, ::-1][:, :8]); Z[0] = 0
k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
NU = cu.max() + 1
n = np.bincount(cu, minlength=NU)
heavy = np.flatnonzero(n >= 50)
half = rng.random(len(cu)) < 0.5
def mean_of(mask):
    S = np.zeros((NU, 8)); np.add.at(S, cu[mask], Z[ci[mask]])
    return S / np.maximum(np.bincount(cu[mask], minlength=NU), 1)[:, None]
T, T1, T2 = mean_of(np.ones(len(cu), bool))[heavy], mean_of(half)[heavy], mean_of(~half)[heavy]
print(f"{len(heavy)} users with >= 50 clicks; taste norm (coherence) mean {np.linalg.norm(T, axis=1).mean():.3f}", flush=True)
def kmeans(A, k, it=60, starts=5):
    best, bestC = -1, None
    for _ in range(starts):
        C = A[rng.choice(len(A), k, replace=False)]
        for _ in range(it):
            a = np.argmax(A @ C.T, 1)
            C = unit(np.stack([A[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)]))
        s = np.max(A @ C.T, 1).mean()
        if s > best: best, bestC = s, C
    return best, bestC
# Gaussian surrogate: truth ~ N(mean, cov of the real taste minus the half-split noise), halves = truth + noise
noise_cov = np.cov((T1 - T2).T) / 4           # var of a half-mean around the full mean ~ var(T1 - T2) / 4 ... per half x2
Ct = np.cov(T.T) - noise_cov
w, V = np.linalg.eigh(Ct); Ct = (V * np.maximum(w, 1e-9)) @ V.T
G = rng.multivariate_normal(T.mean(0), Ct, len(T))
G1 = G + rng.multivariate_normal(np.zeros(8), 2 * noise_cov, len(T)) * 1.0
G2 = G + rng.multivariate_normal(np.zeros(8), 2 * noise_cov, len(T)) * 1.0
Gf = G + rng.multivariate_normal(np.zeros(8), noise_cov, len(T))
print(f"signal var share of the taste estimate: {np.trace(Ct) / np.trace(np.cov(T.T)):.3f}", flush=True)
for kk in (4, 8, 16, 32):
    sr, Cr = kmeans(unit(T), kk); sg, Cg = kmeans(unit(Gf), kk)
    ar = np.mean(np.argmax(unit(T1) @ Cr.T, 1) == np.argmax(unit(T2) @ Cr.T, 1))
    ag = np.mean(np.argmax(unit(G1) @ Cg.T, 1) == np.argmax(unit(G2) @ Cg.T, 1))
    print(f"k {kk:2d}: k-means mean cos real {sr:.4f} gaussian {sg:.4f} | split-half same centre real {ar:.3f} gaussian {ag:.3f}",
          flush=True)
# radial structure: are taste DIRECTIONS uniform? excess kurtosis of whitened coordinates and of the norm
Xw = (T - T.mean(0)) @ np.linalg.inv(np.linalg.cholesky(np.cov(T.T))).T
Gw = (Gf - Gf.mean(0)) @ np.linalg.inv(np.linalg.cholesky(np.cov(Gf.T))).T
print(f"mean |excess kurtosis| of whitened axes: real {np.mean(np.abs((Xw ** 4).mean(0) - 3)):.3f} "
      f"gaussian {np.mean(np.abs((Gw ** 4).mean(0) - 3)):.3f}", flush=True)
