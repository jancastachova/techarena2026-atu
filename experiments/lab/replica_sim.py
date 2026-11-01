# Run from the repository root:
#   .venv/bin/python experiments/lab/replica_sim.py --scale 1.0 [--out data/cache/replica_s1.0.npz]
# then the core MF on the synthetic data (same harness as every MF check, days 0-5 -> day 6):
#   .venv/bin/python experiments/13_retune_7days/retune.py --stage tune --base data/cache/replica_s1.0.npz \
#       --cutoff 7 --tune base --tune-seeds 3
#
# HOW FAR IS OUR ESTIMATOR FROM THE BEST POSSIBLE? A replica of the simulator built from our own fitted
# model, with the clicks of every training event regenerated on the real timeline (real users, events,
# timestamps, candidate sets and warm-up clicks). Truth:
#   item vector phi_i, bias b_i     MF trained on days 0-6 (22_vectors W_test; item-bias term of MF30's test rows)
#   user taste theta_u              unit(sqrt-set mean of phi over the user's real clicks) x R x scale
#   click propensity p_u            drawn from the user's Beta posterior (prior mean 0.63, sd 0.14; DATASET.md)
#   repeat bonus                    +log(4.2) for a candidate already among the user's (simulated) clicks
#   choice                          one of the 10 candidates or no click: softmax with a no-click logit a_u,
#                                   a_u set so the user's expected click share is p_u
# The scale is calibrated so that our MF scores on the synthetic data what it scores on the real data
# (0.8175, 3 seeds); then the oracle AUC (true probabilities) on day 6 is the ceiling for this world.
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--repeat", type=float, default=float(np.log(4.2)))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--proj", type=int, default=0, help="truth item vectors projected onto their top-k principal directions")
ap.add_argument("--act-pow", type=float, default=0.0,
                help="taste norm x (user's clicks / median)^p, mean norm kept (lab/replica_cells.py: heavy users more taste-driven)")
ap.add_argument("--item-norm", type=float, default=0.0,
                help="truth item norms pulled toward their mean: |phi| -> |phi|^(1-q) (q = 1: equal norms; the MF shrinks rare items)")
ap.add_argument("--out")
ap.add_argument("--truth-out", help="write theta, phi, item bias, propensity logit and true probabilities here")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
BASE = ROOT / "data/cache/13_base_c7_9c47a117c762.npz"
fit, test, _, n_users, n_items = e13.load_base(str(BASE))
auc = e13.roc_auc
rng = np.random.default_rng(a.seed)
T0 = e13.TRAIN_START

# ---- truth
PHI = np.load(ROOT / "data/cache/22_vectors_c7.npz")["W_test"].astype(np.float64)
if a.proj:                                   # an a.proj-dim truth (lab/geometry.py: ~8 strong directions), same mean norm
    _nrm0 = np.linalg.norm(PHI[1:], axis=1).mean()
    _, _U = np.linalg.eigh(PHI[1:].T @ PHI[1:]); _U = _U[:, ::-1][:, :a.proj]
    PHI = (PHI @ _U) @ _U.T
    PHI *= _nrm0 / np.linalg.norm(PHI[1:], axis=1).mean()
if a.item_norm:
    _n = np.linalg.norm(PHI[1:], axis=1, keepdims=True); _m = _n.mean()
    PHI[1:] *= (np.maximum(_n, 1e-9) ** (1 - a.item_norm)) / np.maximum(_n, 1e-9)
    PHI[1:] *= _m / np.linalg.norm(PHI[1:], axis=1).mean()
bsum = np.zeros(n_items + 1); bcnt = np.zeros(n_items + 1)
for f in ("13_mf_c7_bb1fcb0a9c89.npz", "15_mf_c7_46f30c940a6a.npz"):
    z = np.load(ROOT / "data/cache" / f)
    np.add.at(bsum, test["item"], z["mf_test"][:, 1]); np.add.at(bcnt, test["item"], 1)
    d6 = np.repeat(fit["day"] == 6, 10)
    np.add.at(bsum, fit["item"][d6], 0.01 * z["mf_fit"][d6, 1]); np.add.at(bcnt, fit["item"][d6], 0.01)
B_ITEM = np.where(bcnt > 0, bsum / np.maximum(bcnt, 1e-9), np.nan)
B_ITEM = np.where(np.isnan(B_ITEM), np.nanmean(B_ITEM), B_ITEM)

cu, ci, cts = fit["clk_user"].astype(np.int64), fit["clk_item"].astype(np.int64), fit["clk_ts"].astype(np.int64)
nu = int(max(cu.max(), fit["user"].max(), test["user"].max())) + 1
pairs, cnt = np.unique(cu * (n_items + 1) + ci, return_counts=True)
pu, pi = pairs // (n_items + 1), pairs % (n_items + 1)
TH = np.zeros((nu, PHI.shape[1])); wsum = np.zeros(nu)
np.add.at(TH, pu, PHI[pi] * np.sqrt(cnt)[:, None]); np.add.at(wsum, pu, np.sqrt(cnt))
has = wsum > 0
TH[has] /= wsum[has, None]
nrm = np.linalg.norm(TH, axis=1)
heavy = np.bincount(cu, minlength=nu) >= 20
R = np.median(nrm[heavy])
TH[has] = TH[has] / np.maximum(nrm[has, None], 1e-12) * R
TH[~has] = TH[rng.choice(np.flatnonzero(has), (~has).sum())]      # no clicks at all: someone's taste
if a.act_pow:
    _nc = np.maximum(np.bincount(cu, minlength=nu), 1).astype(np.float64)
    _f = (_nc / np.median(_nc[has])) ** a.act_pow
    TH *= (_f / _f[has].mean())[:, None]
TH *= a.scale

ylab_ev = fit["label"].reshape(-1, 10).max(1)
n_ev = np.bincount(fit["user"], minlength=nu); n_ck = np.bincount(fit["user"], ylab_ev, nu)
al, be = 6.87, 4.03                                                # Beta(mean 0.63, sd 0.14)
P_U = rng.beta(al + n_ck, be + n_ev - n_ck)

# ---- static utilities of every training row
items = fit["item"].reshape(-1, 10).astype(np.int64); eu = fit["user"].astype(np.int64)
U_STAT = np.einsum("ed,ekd->ek", TH[eu], PHI[items]) + B_ITEM[items]
S_ev = np.exp(U_STAT).sum(1)
med_S = np.zeros(nu)
o = np.argsort(eu, kind="stable"); su = eu[o]; bounds = np.searchsorted(su, np.arange(nu + 1))
for u in np.flatnonzero(n_ev):
    med_S[u] = np.median(S_ev[o[bounds[u]:bounds[u + 1]]])
A_U = np.log(np.maximum(med_S, 1e-12) * (1 - P_U) / P_U)

# ---- simulate in time order
warm = cts < T0
clicked = {}
for u, i in zip(cu[warm], ci[warm]):
    clicked.setdefault(int(u), set()).add(int(i))
order = np.lexsort((np.arange(len(eu)), fit["ts"]))
LABEL = np.zeros((len(eu), 10), np.int8); PTRUE = np.zeros((len(eu), 10))
draws = rng.random(len(eu))
for e in order:
    u = int(eu[e]); cs = clicked.get(u)
    ut = U_STAT[e].copy()
    if cs:
        ut += a.repeat * np.array([int(x) in cs for x in items[e]])
    ex = np.exp(ut); den = ex.sum() + np.exp(A_U[u]); p = ex / den
    PTRUE[e] = p
    c = np.searchsorted(np.cumsum(p), draws[e])
    if c < 10:
        LABEL[e, c] = 1
        clicked.setdefault(u, set()).add(int(items[e, c]))

lab = LABEL.ravel()
rep = np.zeros_like(lab, dtype=bool)
print(f"scale {a.scale}: row CTR {lab.mean():.4f} (real 0.0629)  event click share {LABEL.max(1).mean():.3f} (real 0.629)", flush=True)

# ---- new click log and histories
ce = np.flatnonzero(LABEL.max(1))
new_u = np.r_[cu[warm], eu[ce]]; new_i = np.r_[ci[warm], items[ce, LABEL[ce].argmax(1)]]
new_t = np.r_[cts[warm], fit["ts"][ce].astype(np.int64)]
syn = dict(fit); syn["label"] = lab; syn["clk_user"], syn["clk_item"], syn["clk_ts"] = new_u, new_i, new_t
syn["hist"], syn["hlen"] = e13.long_history(fit, syn, 20)
tst = dict(test); tst["hist"], tst["hlen"] = e13.long_history(test, syn, 20)

# repeat CTR for the sanity table
H100, n100 = e13.long_history(fit, syn, 100)
inh = np.zeros((len(eu), 10), bool)
for j in range(100):
    ok = n100 > j
    inh[ok] |= items[ok] == H100[ok, j][:, None]
print(f"  repeat candidates {inh.mean():.4f} (real 0.026), clicked {LABEL[inh].mean():.3f} (real 0.293), "
      f"others {LABEL[~inh].mean():.4f} (real 0.057)", flush=True)

# ---- oracle on day 6
d6 = fit["day"] == 6
pt, y6 = PTRUE[d6].ravel(), LABEL[d6].ravel()
yev = LABEL[d6].max(1)
print(f"  ORACLE day 6: global AUC {auc(pt, y6):.5f}  event level {auc(PTRUE[d6].sum(1), yev):.5f}  "
      f"within event {np.mean([auc(PTRUE[d6][k], LABEL[d6][k]) for k in np.flatnonzero(yev)[:20000]]):.4f}", flush=True)
hl6 = np.repeat(np.minimum(n100[d6], 100), 10)
# user-side loss: TRUE item vectors and biases, user = sqrt-set mean of the true vectors of the known
# clicks (as the MF builds it), no propensity / repeat terms; and the same with the true taste
H6, n6 = H100[d6], n100[d6].astype(int)
it6 = items[d6]
hv = np.zeros((len(H6), PHI.shape[1]))
for r_ in range(len(H6)):
    if n6[r_]:
        u_, c_ = np.unique(H6[r_, :n6[r_]], return_counts=True)
        hv[r_] = (PHI[u_] * np.sqrt(c_)[:, None]).sum(0) / np.sqrt(c_).sum()
est = (np.einsum("ed,ekd->ek", hv, PHI[it6]) + B_ITEM[it6]).ravel()
tru = (np.einsum("ed,ekd->ek", TH[eu[d6]], PHI[it6]) + B_ITEM[it6]).ravel()
print(f"  true items, taste = pooled true vectors of known clicks: {auc(est, y6):.5f}   "
      f"true taste (no propensity/repeat): {auc(tru, y6):.5f}", flush=True)
for lo, hi in [(1, 5), (6, 20), (21, 100)]:
    k = (hl6 >= lo) & (hl6 <= hi)
    print(f"    known clicks {lo}-{hi}: oracle {auc(pt[k], y6[k]):.5f}  pooled-true-vectors {auc(est[k], y6[k]):.5f}", flush=True)
if a.truth_out:
    np.savez(a.truth_out, theta=TH.astype(np.float32), phi=PHI.astype(np.float32), b_item=B_ITEM.astype(np.float32),
             a_user=A_U.astype(np.float32), p_true=PTRUE.ravel().astype(np.float32), label=lab)
    print(f"  wrote {a.truth_out}", flush=True)
if a.out:
    z = np.load(BASE)
    out = {k: z[k] for k in z.files}
    for k, v in syn.items():
        out["tr_" + k] = v
    for k, v in tst.items():
        out["ev_" + k] = v
    out["oracle_p"] = PTRUE.ravel().astype(np.float32)
    np.savez(a.out, **out)
    print(f"  wrote {a.out}", flush=True)
