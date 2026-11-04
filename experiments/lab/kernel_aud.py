# Run from the repository root: .venv/bin/python experiments/lab/kernel_aud.py [--day 6] [--seeds 3]
# KERNEL AUDIENCE (2026-11-03). The audience view (26, soft_aud = unit(A - 0.5 B)) is LINEAR: the clickers' mean taste minus
# half the shown users' mean taste, dotted with the user's taste. Its local version asks directly "when this item was shown
# to users whose taste is close to this user's, how often was it clicked?":
#     f(u, i) = (sum_v K(u, v) y_vi + alpha * ctr_i) / (sum_v K(u, v) + alpha),   K = exp(kappa * (cos(t_u, t_v) - 1))
# over the impressions (v, i, y_vi) before day D, t_v = the unit 8-dim projected taste of v's clicks before D with v's
# own clicks on i removed, t_u = the user's known clicks (up to 100) as of the event. kappa = 0 is the item's click rate;
# small kappa is the linear audience; larger kappa is a nearest-neighbour vote. A different estimator of the same item
# direction for rare items (~140 impressions each), nonparametric in taste space.
# Single-score AUC on day D (plain / eval-weighted), MF vectors of days < D (38_vectors), averaged over seeds; then the
# same scores ranked inside each event (they carry item popularity, the cosines do not).
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--day", type=int, default=6)
ap.add_argument("--out", default="")
ap.add_argument("--noself", action="store_true", help="leave the user's own past impressions of the item out of the kernel")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
spec2 = importlib.util.spec_from_file_location("aud", ROOT / "experiments/26_audience/audience.py")
aud = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(aud)
aud.UNIT_CLICKERS = True
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
D = a.day
evd = np.flatnonzero(fit["day"] == D); items = fit["item"].reshape(-1, 10)[evd].astype(np.int64)
y = fit["label"].reshape(-1, 10)[evd].ravel().astype(float)
H, n = e13.long_history(fit, fit, 100); H, n = H[evd].astype(np.int64), n[evd].astype(int)
msk = (np.arange(100)[None] < n[:, None]).astype(np.float64)[:, :, None]
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


def unit(X): return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


# impressions before day D: (user, item, clicked)
pre = np.repeat(fit["day"] < D, 10)
iu = np.repeat(fit["user"].astype(np.int64), 10)[pre]; ii = fit["item"][pre].astype(np.int64); iy = fit["label"][pre].astype(float)
kc = fit["clk_ts"] < e13.TRAIN_START + D * e13.DAY
cu, ci = fit["clk_user"][kc].astype(np.int64), fit["clk_item"][kc].astype(np.int64)
NI = int(max(n_items + 1, ii.max() + 1, ci.max() + 1))
pk, pc = np.unique(cu * NI + ci, return_counts=True)                 # clicks of user v on item i before D
q = np.searchsorted(pk, iu * NI + ii); q = np.minimum(q, len(pk) - 1)
own = np.where(pk[q] == iu * NI + ii, pc[q], 0).astype(float)         # v's clicks on i, removed from v's taste
ctr_i = (np.bincount(ii, iy, minlength=NI) + 0.06 * 20) / (np.bincount(ii, minlength=NI) + 20)
o = np.argsort(ii, kind="stable"); iu, ii, iy, own = iu[o], ii[o], iy[o], own[o]
start = np.searchsorted(ii, np.arange(NI)); stop = np.searchsorted(ii, np.arange(NI), side="right")
rows_item = items.ravel()
row_user = np.repeat(fit["user"][evd].astype(np.int64), 10)
ro = np.argsort(rows_item, kind="stable")
rstart = np.searchsorted(rows_item[ro], np.arange(NI)); rstop = np.searchsorted(rows_item[ro], np.arange(NI), side="right")
KAPPAS = (2.0, 5.0, 10.0, 20.0)
ALPHA = 3.0
acc = {k: np.zeros(len(y)) for k in ["own cos", "aud cos"] + [f"kernel k{k:g}" for k in KAPPAS]}
nu = int(max(cu.max(), iu.max(), fit["user"].max())) + 1
for s in range(a.seeds):
    W = np.load(ROOT / f"data/cache/38_vectors_c7_s{s}.npz")[f"W_{D}"].astype(np.float64)
    _, Ue = np.linalg.eigh(W[1:].T @ W[1:]); P = Ue[:, ::-1][:, :8]
    Z = unit(W @ P); Z[0] = 0
    Zp = np.zeros((NI, 8)); Zp[:len(Z)] = Z
    hv = unit((Zp[H] * msk).sum(1))                                   # (events, 8) user taste as of the event
    tu = np.repeat(hv, 10, axis=0)                                    # per test row
    acc["own cos"] += np.einsum("rd,rd->r", tu, Zp[rows_item])
    A, na, B = aud.audience(fit, Z, D)
    Au = unit(A - 0.5 * B); Au[0] = 0; Ap = np.zeros((NI, 8)); Ap[:len(Au)] = Au
    acc["aud cos"] += np.einsum("rd,rd->r", tu, Ap[rows_item])
    S = np.zeros((nu, 8)); np.add.at(S, cu, Zp[ci])
    tv = unit(S[iu] - own[:, None] * Zp[ii])                          # impression users' tastes, own clicks on i removed
    out = {k: np.zeros(len(y)) for k in KAPPAS}
    for it in np.flatnonzero((rstop > rstart) & (stop > start)):
        rr = ro[rstart[it]:rstop[it]]
        sim = tu[rr] @ tv[start[it]:stop[it]].T - 1.0                 # (rows, impressions)
        yy = iy[start[it]:stop[it]]
        if a.noself:                                                  # repeats / re-shows are LightGBM's job
            sim[row_user[rr][:, None] == iu[start[it]:stop[it]][None]] = -np.inf
        for k in KAPPAS:
            K = np.exp(k * sim)
            out[k][rr] = (K @ yy + ALPHA * ctr_i[it]) / (K.sum(1) + ALPHA)
    for k in KAPPAS:
        miss = np.ones(len(y), bool); miss[ro[np.concatenate([np.arange(rstart[i_], rstop[i_]) for i_ in
                                                             np.flatnonzero((rstop > rstart) & (stop > start))])]] = False
        out[k][miss] = ctr_i[rows_item[miss]]
        acc[f"kernel k{k:g}"] += out[k]
    print(f"seed {s} done", flush=True)
acc["item ctr"] = ctr_i[rows_item]
ev_of = np.repeat(np.arange(len(evd)), 10)
print(f"day {D}, {a.seeds} seeds; single-score AUC plain / eval-weighted; within-event AUC (rank inside the event)")
for k, v in acc.items():
    r = v.reshape(-1, 10); rk = (r.argsort(1).argsort(1) / 9.0).ravel()
    clicked = y.reshape(-1, 10).sum(1) > 0
    m = np.repeat(clicked, 10)
    print(f"  {k:14s} {e13.roc_auc(v, y):.5f}  {wauc(v, y, w_ev):.5f}   within {e13.roc_auc(rk[m] + 1e-6 * v[m], y[m]):.5f}",
          flush=True)
if a.out:
    np.savez_compressed(a.out, evd=evd, **{k.replace(" ", "_"): v for k, v in acc.items()})
    print("wrote", a.out)
