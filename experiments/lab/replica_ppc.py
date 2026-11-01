# Run from the repository root: .venv/bin/python experiments/lab/replica_ppc.py
# POSTERIOR PREDICTIVE CHECKS: the calibrated replica (lab/replica_sim.py --scale 1.42 --repeat 0) holds exactly
# the mechanisms we model (fixed taste x item vectors + item bias + per-user propensity) and nothing else. A
# statistic that differs between the real data and the replica points at a mechanism we are missing.
# Train-period clicks only (the replica regenerates those; warm-up clicks are shared), item vectors W_test.
#   1. sequence: cosine between a click and the user's click k steps earlier (k = 1, 2, 5, 10), and a
#      within-user shuffle of the same clicks (the no-order null of each world)
#   2. popularity: mean log popularity of clicked items, and the share of clicks that are repeats
#   3. timing: cosine between consecutive clicks by the time gap between them
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
T0 = e13.TRAIN_START
W = np.load(ROOT / "data/cache/22_vectors_c7.npz")["W_test"].astype(np.float64)
Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
rng = np.random.default_rng(0)

def clicks(path):
    z = np.load(path)
    u, i, t = z["tr_clk_user"].astype(np.int64), z["tr_clk_item"].astype(np.int64), z["tr_clk_ts"].astype(np.int64)
    o = np.lexsort((t, u)); return u[o], i[o], t[o]

def stats(name, path):
    u, i, t = clicks(path)
    pop = np.log1p(np.bincount(i[t >= T0], minlength=len(W)))
    tr = t >= T0
    out = [f"{name:8s}"]
    # lagged cosine within user, train-period click pairs only
    for k in (1, 2, 5, 10):
        a = np.arange(k, len(u)); b = a - k
        ok = (u[a] == u[b]) & tr[a] & tr[b]
        out.append(f"lag{k} {np.mean(np.einsum('nd,nd->n', Wn[i[a[ok]]], Wn[i[b[ok]]])):.4f}")
    # within-user shuffle null (train clicks permuted inside each user)
    ut = u[tr]; it_ = i[tr].copy()
    start = np.searchsorted(ut, ut, "left"); perm = np.arange(len(ut))
    keys = rng.random(len(ut)) + ut * 2.0
    sh = it_[np.argsort(keys, kind="stable")]
    a = np.arange(1, len(ut)); ok = ut[a] == ut[a - 1]
    out.append(f"shuffled lag1 {np.mean(np.einsum('nd,nd->n', Wn[sh[a[ok]]], Wn[sh[a[ok] - 1]])):.4f}")
    # exact repeats of the previous click / of any earlier train click
    a = np.arange(1, len(u)); ok = (u[a] == u[a - 1]) & tr[a] & tr[a - 1]
    out.append(f"same item as previous {np.mean(i[a[ok]] == i[a[ok] - 1]):.4f}")
    out.append(f"click log-pop {pop[i[tr]].mean():.3f}")
    print("  ".join(out), flush=True)
    # cosine of consecutive clicks by gap
    g = (t[a[ok]] - t[a[ok] - 1]) / 3600
    c = np.einsum('nd,nd->n', Wn[i[a[ok]]], Wn[i[a[ok] - 1]])
    print("          by gap (h): " + "  ".join(f"<{hi}: {c[(g >= lo) & (g < hi)].mean():.4f} (n={((g >= lo) & (g < hi)).sum()})"
                                          for lo, hi in [(0, 0.5), (0.5, 3), (3, 12), (12, 48), (48, 1000)]), flush=True)


# FAIR version: each world's own MF vectors trained on its days 0-5 (real: 22_vectors W_6, replica:
# replica_vec_s0 W_6), statistics on DAY-6 clicks only (unseen by both MFs), against all earlier clicks.
def fair(name, path, vec):
    global Wn
    Wv = np.load(vec)["W_6"].astype(np.float64); Wn = Wv / np.maximum(np.linalg.norm(Wv, axis=1, keepdims=True), 1e-9)
    u, i, t = clicks(path)
    d6 = t >= T0 + 6 * 86400
    out = [f"{name:8s} day-6 clicks {d6.sum()}"]
    for k in (1, 2, 3, 5, 10, 20):
        a = np.flatnonzero(d6); a = a[a >= k]; b = a - k
        ok = u[a] == u[b]
        out.append(f"lag{k} {np.mean(np.einsum('nd,nd->n', Wn[i[a[ok]]], Wn[i[b[ok]]])):.4f}")
    a = np.flatnonzero(d6); a = a[a >= 1]; ok = u[a] == u[a - 1]
    out.append(f"same item as previous {np.mean(i[a[ok]] == i[a[ok] - 1]):.4f}")
    for k in (2, 5):
        a2 = np.flatnonzero(d6); a2 = a2[a2 >= k]; ok2 = u[a2] == u[a2 - k]
        out.append(f"same as lag{k} {np.mean(i[a2[ok2]] == i[a2[ok2] - k]):.4f}")
    print("  ".join(out), flush=True)

fair("real", ROOT / "data/cache/13_base_c7_9c47a117c762.npz", ROOT / "data/cache/22_vectors_c7.npz")
fair("replica", ROOT / "data/cache/replica_s1.42_r0.npz", ROOT / "data/cache/replica_vec_s0.npz")
