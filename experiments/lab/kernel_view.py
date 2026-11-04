# Run from the repository root: .venv/bin/python experiments/lab/kernel_view.py --out data/cache/lab_kernel_view.npz
# KERNEL AUDIENCE as a feature group for the lean-export lab (lab/lgb_choice.py --extra). See lab/kernel_aud.py: for a row
# (user u, item i) on day d,
#     f_kappa(u, i) = (sum_v K(u, v) y_vi + 3 ctr_i) / (sum_v K(u, v) + 3),   K = exp(kappa * (cos(t_u, t_v) - 1)),
# over the impressions (v, i, y_vi) of fit days < d, t_v = unit 8-dim projected taste of v's clicks before d minus v's own
# clicks on i, t_u = the user's known clicks (up to 100) as of the event; MF vectors W_d of an MF trained on days < d
# (38_vectors, the time-forward rule), f averaged over seeds. Columns, for kappa 2 / 5 / 10, with ("self") and without
# ("noself") the user's own past impressions of the item: log-odds of f, and the same minus its event mean.
# Rows: fit rows with day >= 2 in file order (the lean export's row order). Day-d statistics use labels of days < d only.
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", required=True)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
KAPPAS = (2.0, 5.0, 10.0)
ALPHA = 3.0
H_all, n_all = e13.long_history(fit, fit, 100)
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
NI = int(max(n_items + 1, fit["item"].max() + 1, fit["clk_item"].max() + 1))
nu = int(max(fit["clk_user"].max(), fit["user"].max())) + 1


def unit(X): return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def day_scores(D, Zp):
    """Kernel scores (self, noself) x kappa for every row of the events on day D."""
    evd = np.flatnonzero(fit["day"] == D)
    items = fit["item"].reshape(-1, 10)[evd].astype(np.int64).ravel()
    H, n = H_all[evd].astype(np.int64), n_all[evd].astype(int)
    msk = (np.arange(100)[None] < n[:, None]).astype(np.float64)[:, :, None]
    tu = np.repeat(unit((Zp[H] * msk).sum(1)), 10, axis=0)
    ruser = np.repeat(fit["user"][evd].astype(np.int64), 10)
    pre = np.repeat(fit["day"] < D, 10)
    iu = np.repeat(fit["user"].astype(np.int64), 10)[pre]; ii = fit["item"][pre].astype(np.int64)
    iy = fit["label"][pre].astype(float)
    kc = fit["clk_ts"] < e13.TRAIN_START + D * e13.DAY
    cu, ci = fit["clk_user"][kc].astype(np.int64), fit["clk_item"][kc].astype(np.int64)
    pk, pc = np.unique(cu * NI + ci, return_counts=True)
    q = np.minimum(np.searchsorted(pk, iu * NI + ii), len(pk) - 1)
    own = np.where(pk[q] == iu * NI + ii, pc[q], 0).astype(float)
    ctr = (np.bincount(ii, iy, minlength=NI) + 0.06 * 20) / (np.bincount(ii, minlength=NI) + 20)
    o = np.argsort(ii, kind="stable"); iu, ii, iy, own = iu[o], ii[o], iy[o], own[o]
    st, sp = np.searchsorted(ii, np.arange(NI)), np.searchsorted(ii, np.arange(NI), side="right")
    S = np.zeros((nu, Zp.shape[1])); np.add.at(S, cu, Zp[ci])
    tv = unit(S[iu] - own[:, None] * Zp[ii])
    ro = np.argsort(items, kind="stable")
    rst, rsp = np.searchsorted(items[ro], np.arange(NI)), np.searchsorted(items[ro], np.arange(NI), side="right")
    out = np.repeat(ctr[items][:, None], 2 * len(KAPPAS), axis=1)    # rows whose item was never shown before D
    for it in np.flatnonzero((rsp > rst) & (sp > st)):
        rr = ro[rst[it]:rsp[it]]
        sim = tu[rr] @ tv[st[it]:sp[it]].T - 1.0
        yy = iy[st[it]:sp[it]]
        selfm = ruser[rr][:, None] == iu[st[it]:sp[it]][None]
        for j, k in enumerate(KAPPAS):
            K = np.exp(k * sim)
            out[rr, j] = (K @ yy + ALPHA * ctr[it]) / (K.sum(1) + ALPHA)
            K[selfm] = 0
            out[rr, len(KAPPAS) + j] = (K @ yy + ALPHA * ctr[it]) / (K.sum(1) + ALPHA)
    return evd, out


F = np.zeros((len(rows), 2 * len(KAPPAS)))
pos = np.full(len(fit["day"]) * 10, -1); pos[rows] = np.arange(len(rows))
for D in range(2, 7):
    acc = None
    for s in range(a.seeds):
        W = np.load(ROOT / f"data/cache/38_vectors_c7_s{s}.npz")[f"W_{D}"].astype(np.float64)
        _, Ue = np.linalg.eigh(W[1:].T @ W[1:]); P = Ue[:, ::-1][:, :8]
        Z = unit(W @ P); Z[0] = 0
        Zp = np.zeros((NI, 8)); Zp[:len(Z)] = Z
        evd, sc = day_scores(D, Zp)
        acc = sc if acc is None else acc + sc
    r = (evd[:, None] * 10 + np.arange(10)[None]).ravel()
    F[pos[r]] = acc / a.seeds
    print(f"day {D} done", flush=True)
L = lg(F)
R = (L.reshape(-1, 10, L.shape[1]) - L.reshape(-1, 10, L.shape[1]).mean(1, keepdims=True)).reshape(L.shape)
names = [f"kern_{v}_k{k:g}" for v in ("self", "noself") for k in KAPPAS]
X = np.concatenate([L, R], 1).astype(np.float32)
names = names + [n + "_rel" for n in names]
np.savez(a.out, X=X, names=np.array(names), rows=rows)
print("wrote", a.out, X.shape)
