# Run from the repository root: .venv/bin/python experiments/lab/geometry2.py [--seeds 5]
# ONE BETTER ITEM VECTOR (2026-10-30). The two views that carry 75% of SET38's LightGBM gain are separate estimates of the
# same true item direction: the item's own MF vector projected on the 8 strong directions (cosproj8) and its AUDIENCE, the
# mean unit taste of its clickers minus 0.5 x that of the users it was shown to (soft_aud / audc05). The own vector is noisy
# for rare items; the audience pools the clickers' whole histories. Combined at the VECTOR level, with the audience weighted
# more for items with few clicks, they may give one score stronger than either -- a representation change, the kind that
# transferred ~1:1 to the leaderboard (34 cosine +0.0011, 35 projection +0.0013).
# Day 6, item vectors of the MF of days 0-5 (38_vectors, aux 0.3 / 4500 steps), scores averaged over seeds; single-score AUC
# (plain and evaluation-weighted) of cos(user, item) with
#   own        Z_i = unit(P8 W_i)
#   aud        unit(A_i - 0.5 B_i) from the unit projected vectors (unit clicker tastes)
#   mix K g    unit(Z_i * c/(c+K) + g * aud_i * K/(c+K)), c = item clicks before day 6
#   add g      unit(Z_i + g * aud_i)
# user = mean of the item representation over the user's known clicks (up to 100), or of Z ("u=own").
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--day", type=int, default=6)
ap.add_argument("--iterate", type=int, default=0, help="also run R <- unit(Z + g x audience(R)) for this many rounds")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
spec2 = importlib.util.spec_from_file_location("aud", ROOT / "experiments/26_audience/audience.py")
aud = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(aud)
aud.UNIT_CLICKERS = True
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
D = a.day
evd = np.flatnonzero(fit["day"] == D); items = fit["item"].reshape(-1, 10)[evd].astype(np.int64)
y = fit["label"].reshape(-1, 10)[evd].ravel()
H, n = e13.long_history(fit, fit, 100); H, n = H[evd].astype(np.int64), n[evd].astype(int)
msk = (np.arange(100)[None] < n[:, None]).astype(np.float32)[:, :, None]
_, hl_eval = e13.long_history(test, fit, 100)
BK = [0, 5, 20, 50, 100]
bk = lambda h: np.searchsorted(BK, np.asarray(h, int))
s_ev = np.bincount(bk(hl_eval), minlength=5) / len(hl_eval); s_d = np.bincount(bk(n), minlength=5) / len(n)
w_ev = np.repeat((s_ev / np.maximum(s_d, 1e-9))[bk(n)], 10)
def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
c = np.bincount(fit["clk_item"][fit["clk_ts"] < e13.TRAIN_START + D * e13.DAY], minlength=n_items + 1).astype(np.float64)
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
def cos_score(R, U=None):
    U = R if U is None else U
    hv = unit((U[H] * msk).sum(1))
    return np.einsum("ed,ekd->ek", hv, R[items]).ravel()

variants = {"own": None, "aud": None}
for K in (5, 20, 80):
    for g in (0.5, 1.0, 2.0):
        variants[f"mix K{K} g{g}"] = (K, g)
for g in (0.5, 1.0):
    variants[f"add g{g}"] = (None, g)
variants["mix K20 g1.0 u=own"] = (20, 1.0)
acc = {k: 0.0 for k in variants}
for s in range(a.seeds):
    W = np.load(ROOT / f"data/cache/38_vectors_c7_s{s}.npz")[f"W_{D}"].astype(np.float64)
    _, Ue = np.linalg.eigh(W[1:].T @ W[1:]); P = Ue[:, ::-1][:, :8]
    Z = unit(W @ P); Z[0] = 0
    A, na, B = aud.audience(fit, Z, D)
    Au = unit(A - 0.5 * B); Au[0] = 0
    cc = c[:len(Z)][:, None]
    for k, v in variants.items():
        if k == "own": R, U = Z, None
        elif k == "aud": R, U = Au, None
        else:
            K, g = v
            R = unit(Z * cc / (cc + K) + g * Au * K / (cc + K)) if K else unit(Z + g * Au)
            R[0] = 0
            U = Z if k.endswith("u=own") else None
        acc[k] = acc[k] + cos_score(R, U)
    if a.iterate:                        # fixed point: audiences recomputed from the combined representation
        for beta in (0.5, 1.0):
            for g in (1.0, 2.0):
                R = unit(Z + g * Au); R[0] = 0
                for it in range(1, a.iterate + 1):
                    A2, _, B2 = aud.audience(fit, R, D)
                    R = unit(Z + g * unit(A2 - beta * B2)); R[0] = 0
                    k = f"iter{it} beta{beta} g{g}"
                    acc[k] = acc.get(k, 0.0) + cos_score(R)
    print(f"seed {s} done", flush=True)
print(f"day {D}, {a.seeds} seeds; single-score AUC plain / eval-weighted")
for k in list(variants) + [k_ for k_ in acc if k_ not in variants]:
    print(f"  {k:22s} {e13.roc_auc(acc[k], y):.5f}  {wauc(acc[k], y, w_ev):.5f}", flush=True)
