# Run from the repository root: .venv/bin/python experiments/lab/category_view.py --k 9 --rank 8 --out data/cache/lab_cat9.npz
# 2026-11-04: CATEGORY VIEW. lab/exposure_cluster.py found discrete item categories in EXPOSURE space (mean cosine to the
# k-means centre 0.93 vs 0.58 for a Gaussian), nearly orthogonal to the MF's taste space (canonical correlations <= 0.09).
# Items are partitioned once from the normalised user x item matrix of shown candidates (re-shows of the user's own clicks
# removed). The partition uses NO labels; --prefix D builds it from events before day D only, to check that it does not
# depend on later exposure. Per event, as of the event (strictly earlier events of the user; the slate of the event itself
# is a separate column), per candidate of category c:
#   expo_lift   log of the user's smoothed share of shown candidates in c over the global share
#   click_lift  log of the user's smoothed share of known clicks in c over the global click share
#   cat_ctr     the user's smoothed click rate on candidates of c in earlier events
#   slate_n     candidates of c in this slate; slate_click_mass: slate-level sum of the click lifts (softmax denominator)
#   prof_cos    cosine of the candidate with the user's clicked-item profile in exposure space; expo_cos: with the user's
#               shown-item profile; cat: the category id; cat_conf: the candidate's cosine to its centre
# plus the lifts minus their event means. Rows: fit rows with day >= 2 in file order (the lean export's order), and test rows.
import argparse, importlib.util
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
ROOT = Path.cwd()
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=9); ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--prefix", type=int, default=7, help="build the partition from fit events before this day")
ap.add_argument("--min-impr", type=int, default=50); ap.add_argument("--out", required=True)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
K, R = a.k, a.rank
NF, NT = len(fit["user"]), len(test["user"])
user = np.r_[fit["user"], test["user"]].astype(np.int64); ts = np.r_[fit["ts"], test["ts"]]
item = np.r_[fit["item"], test["item"]].astype(np.int64).reshape(-1, 10)
NI = int(max(n_items + 1, item.max() + 1, fit["clk_item"].max() + 1)); NU = int(max(user.max(), fit["clk_user"].max())) + 1


def unit(X): return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


ck = fit["clk_user"].astype(np.int64) * NI + fit["clk_item"].astype(np.int64)
o = np.lexsort((fit["clk_ts"], ck)); ck_s, ct_s = ck[o], fit["clk_ts"][o]
f1 = np.r_[True, ck_s[1:] != ck_s[:-1]]; pk, pt = ck_s[f1], ct_s[f1]
key = (np.repeat(user, 10) * NI + item.ravel())
q = np.minimum(np.searchsorted(pk, key), len(pk) - 1)
reshow = ((pk[q] == key) & (pt[q] < np.repeat(ts, 10))).reshape(-1, 10)        # candidate already clicked by the user

# ---- partition from exposure
ev_use = np.zeros(NF + NT, bool); ev_use[:NF] = fit["day"] < a.prefix
m = np.repeat(ev_use, 10) & ~reshow.ravel()
M = sp.csr_matrix((np.ones(m.sum(), np.float32), (np.repeat(user, 10)[m], item.ravel()[m])), shape=(NU, NI))
du = np.asarray(M.sum(1)).ravel(); di = np.asarray(M.sum(0)).ravel()
Mn = sp.diags(1 / np.sqrt(np.maximum(du, 1))) @ M @ sp.diags(1 / np.sqrt(np.maximum(di, 1)))
U, S, Vt = svds(Mn.astype(np.float64), k=R + 1); o = np.argsort(-S); S, V = S[o], Vt[o].T
E = V[:, 1:R + 1] / np.sqrt(np.maximum(di, 1))[:, None]                          # item embedding, degree-corrected
good = np.flatnonzero(di >= a.min_impr)
mu = E[good].mean(0); Lw = np.linalg.inv(np.linalg.cholesky(np.cov((E[good] - mu).T))).T
Ew = (E - mu) @ Lw; Ew[di == 0] = 0
Eu = unit(Ew)
rng = np.random.default_rng(0)
best = -1
for s in range(8):
    C = Eu[rng.choice(good, K, replace=False)]
    for _ in range(60):
        lab = np.argmax(Eu[good] @ C.T, 1)
        C = unit(np.stack([Eu[good][lab == j].mean(0) if (lab == j).any() else C[j] for j in range(K)]))
    v = np.max(Eu[good] @ C.T, 1).mean()
    if v > best: best, Cb = v, C
sim = Eu @ Cb.T
cat = np.argmax(sim, 1); conf = sim.max(1)
print(f"partition: {len(good):,} items >= {a.min_impr} impressions, mean cosine to centre {best:.4f}; sizes {np.bincount(cat[di > 0], minlength=K)}", flush=True)
print("item confidence quantiles by impressions:", [(lo, round(float(np.median(conf[(di >= lo) & (di < hi)])), 3)) for lo, hi in
                                                    ((1, 10), (10, 30), (30, 100), (100, 300), (300, 1e9))], flush=True)

# ---- per-event counts, as of the event
H_f, n_f = e13.long_history(fit, fit, 100); H_t, n_t = e13.long_history(test, fit, 100)
H = np.r_[H_f, H_t].astype(np.int64); nh = np.r_[n_f, n_t].astype(int)
msk = np.arange(100)[None] < nh[:, None]
ci = cat[item]                                                                   # (events, 10)
Xe = np.stack([((ci == c) & ~reshow).sum(1) for c in range(K)], 1).astype(np.float64)    # shown in this slate, per category
Ce = np.stack([((cat[H] == c) & msk).sum(1) for c in range(K)], 1).astype(np.float64)    # known clicks per category
lab_ev = np.zeros((NF + NT, 10)); lab_ev[:NF] = fit["label"].reshape(-1, 10)
Ye = np.stack([(lab_ev * (ci == c)).sum(1) for c in range(K)], 1)                # clicks in this slate (0 on test events)
o = np.lexsort((ts, user))
us = user[o]; start = np.r_[True, us[1:] != us[:-1]]; first_idx = np.maximum.accumulate(np.where(start, np.arange(len(us)), 0))


def prior_cum(A):
    """Sum of A over the user's strictly earlier events."""
    c = np.cumsum(A[o], 0); ex = c - A[o]
    base = np.where(first_idx[:, None] > 0, c[np.maximum(first_idx - 1, 0)], 0)
    out = np.zeros_like(A); out[o] = ex - base
    return out


X = prior_cum(Xe); Y = prior_cum(Ye)
fitm = ev_use                                    # global shares from the same prefix as the partition
s_expo = Xe[fitm].sum(0) / Xe[fitm].sum()
s_click = Ye[fitm].sum(0) / Ye[fitm].sum()
r_cat = Ye[fitm].sum(0) / np.maximum(Xe[fitm].sum(0), 1)
print("global exposure share", np.round(s_expo, 3)); print("global click share   ", np.round(s_click, 3)); print("category CTR         ", np.round(r_cat, 4))
AE, AC, AR = 10.0, 5.0, 10.0
lift_e = np.log((X + AE * s_expo) / (X.sum(1, keepdims=True) + AE)) - np.log(s_expo)
lift_c = np.log((Ce + AC * s_click) / (Ce.sum(1, keepdims=True) + AC)) - np.log(s_click)
ctr_u = (Y + AR * r_cat) / (X + AR)
rows_ev = np.arange(NF + NT)[:, None]
fe, fc, fr = lift_e[rows_ev, ci], lift_c[rows_ev, ci], np.log(ctr_u[rows_ev, ci])
sl = (ci[:, :, None] == ci[:, None, :]).sum(2).astype(np.float64)                # candidates of the same category in the slate
mass = (np.exp(fc)).sum(1, keepdims=True) * np.ones((1, 10))
Ec = unit(Ew[H] * msk[:, :, None]).sum(1) if False else (Eu[H] * msk[:, :, None]).sum(1)
prof = unit(Ec)
prof_cos = np.einsum("ed,ekd->ek", prof, Eu[item])
SE = np.zeros((NF + NT, R))
Ser = (Eu[item] * (~reshow)[:, :, None]).sum(1)
SE = prior_cum(Ser); expo_cos = np.einsum("ed,ekd->ek", unit(SE), Eu[item])
rel = lambda v: v - v.mean(1, keepdims=True)
cols = {"cat": ci.astype(np.float64), "cat_conf": conf[item], "cat_expo_lift": fe, "cat_click_lift": fc, "cat_ctr": fr, "cat_slate_n": sl,
        "cat_click_mass": np.log(mass), "cat_prof_cos": prof_cos, "cat_expo_cos": expo_cos,
        "cat_expo_lift_rel": rel(fe), "cat_click_lift_rel": rel(fc), "cat_ctr_rel": rel(fr), "cat_prof_cos_rel": rel(prof_cos), "cat_expo_cos_rel": rel(expo_cos)}
names = [f"{n}_k{K}" for n in cols]
A_ = np.stack(list(cols.values()), 2).astype(np.float32).reshape(-1, len(names))
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
Xf, Xt = A_[rows], A_[NF * 10:]
yv = fit["label"][rows].astype(float); dv = np.repeat(fit["day"], 10)[rows]
for n, j in ((n, list(cols).index(n)) for n in ("cat_expo_lift", "cat_click_lift", "cat_ctr", "cat_prof_cos", "cat_expo_cos")):
    mm = dv == 6
    print(f"  day-6 AUC alone {n:16s} {e13.roc_auc(Xf[mm, j].astype(np.float64), yv[mm]):.5f}", flush=True)
mf_fit = A_[:NF * 10].copy(); mf_fit[np.repeat(fit["day"], 10) < 2] = np.nan        # the ensemble's cache format (days 0-1 unused)
np.savez(a.out, X=Xf, X_test=Xt, names=np.array(names), rows=rows, cat=cat, conf=conf, mf_fit=mf_fit, mf_test=Xt)
print("wrote", a.out, Xf.shape, Xt.shape)
