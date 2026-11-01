# Run from the repository root: .venv/bin/python experiments/lab/replica_item_limit.py
# The information limit on the ITEM side of the calibrated replica. Each item's vector and bias are
# re-fitted from its own impressions of days 0-5 by penalised logistic regression (batched Newton) on
# user tastes, with every other part of each impression's true logit as a fixed offset:
#   (a) TRUE tastes theta_u                          -> what the data allows if users were known
#   (b) the MF's pooled user vectors (seed 0 space)  -> one alternating step from our own estimates
# Day-6 score theta_true . phi_hat + b_hat against the true taste logit (AUC 0.87275), and the same
# correlation-by-popularity table as lab/replica_bench.py (MF: 0.824 overall, 0.49 for 0-3 clicks).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/replica_s1.42_r0.npz"))
T = np.load(ROOT / "data/cache/replica_s1.42_r0_truth.npz")
auc = e13.roc_auc
TH, PHI, BI = T["theta"].astype(np.float64), T["phi"].astype(np.float64), T["b_item"].astype(np.float64)
ptrue = np.clip(T["p_true"].astype(np.float64), 1e-9, 1 - 1e-9)
lab = fit["label"].astype(np.float64)
ev05 = np.flatnonzero(fit["day"] < 6)
rows = (ev05[:, None] * 10 + np.arange(10)).ravel()
it = fit["item"][rows].astype(np.int64); us = np.repeat(fit["user"][ev05].astype(np.int64), 10)
y_tr = lab[rows]
own = np.einsum("nd,nd->n", TH[us], PHI[it]) + BI[it]
offset = np.log(ptrue[rows] / (1 - ptrue[rows])) - own          # everything but the item's own term
o = np.argsort(it, kind="stable"); it, us, y_tr, offset = it[o], us[o], y_tr[o], offset[o]
starts = np.searchsorted(it, np.arange(n_items + 2))

def fit_items(X_user, lam, iters=15, chunk=400):
    D = X_user.shape[1] + 1
    out_w = np.zeros((n_items + 1, D - 1)); out_b = np.zeros(n_items + 1)
    for c0 in range(1, n_items + 1, chunk):
        c1 = min(c0 + chunk, n_items + 1)
        a, b = starts[c0], starts[c1]
        if a == b:
            continue
        X = np.c_[X_user[us[a:b]], np.ones(b - a)]
        g_id = it[a:b] - c0; ng = c1 - c0
        w = np.zeros((ng, D))
        reg = np.full(D, lam); reg[-1] = 0.1
        present, seg = np.unique(g_id, return_index=True)          # g_id is sorted (impressions sorted by item)
        for _ in range(iters):
            z = np.einsum("nd,nd->n", X, w[g_id]) + offset[a:b]
            p = 1 / (1 + np.exp(-z))
            G = np.zeros((ng, D)); G[present] = np.add.reduceat(X * (y_tr[a:b] - p)[:, None], seg, axis=0); G -= reg * w
            Hm = np.zeros((ng, D, D))
            Hm[present] = np.add.reduceat((X * (p * (1 - p))[:, None])[:, :, None] * X[:, None, :], seg, axis=0)
            Hm += np.eye(D) * reg
            step = np.linalg.solve(Hm, G[:, :, None])[:, :, 0]
            nrm = np.linalg.norm(step, axis=1, keepdims=True)
            w += step * np.minimum(1.0, 1.0 / np.maximum(nrm, 1e-12))   # damped: at most norm 1 per step
        out_w[c0:c1] = w[:, :-1]; out_b[c0:c1] = w[:, -1]
    return out_w, out_b

z = np.load(ROOT / "data/cache/replica_vec_s0.npz")
evd = z["ev_6"]; items6 = fit["item"].reshape(-1, 10)[evd].astype(np.int64); eu6 = fit["user"][evd].astype(np.int64)
y6 = lab.reshape(-1, 10)[evd].ravel()
true_logit = (np.einsum("ed,ekd->ek", TH[eu6], PHI[items6]) + BI[items6]).ravel()
k = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
pop = np.bincount(fit["clk_item"][k], minlength=n_items + 1); cpop = pop[items6].ravel()
PB = [("0-3", cpop <= 3), ("4-10", (cpop >= 4) & (cpop <= 10)), ("11-30", (cpop >= 11) & (cpop <= 30)), ("31+", cpop >= 31)]
def report(name, s):
    rows_ = "  ".join(f"{g} {auc(s[m], y6[m]):.4f}/{np.corrcoef(s[m], true_logit[m])[0, 1]:.3f}" for g, m in PB)
    print(f"{name:40s} AUC {auc(s, y6):.5f}  corr {np.corrcoef(s, true_logit)[0, 1]:.4f} | {rows_}", flush=True)
report("true taste logit", true_logit)
sd = PHI[1:].std()
for lam in (1.0 / sd ** 2 / 10, 1.0 / sd ** 2):
    W, b = fit_items(TH, lam)
    report(f"(a) true tastes, lam {lam:.0f}", (np.einsum("ed,ekd->ek", TH[eu6], W[items6]) + b[items6]).ravel())

# (b) users = the MF's pooled vectors (seed 0): every user's clicks before day 6
cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
Wmf = z["W_6"].astype(np.float64)
nu = TH.shape[0]
U = np.zeros((nu, Wmf.shape[1])); np.add.at(U, cu, Wmf[ci]); U /= np.maximum(np.bincount(cu, minlength=nu), 1)[:, None]
sd_mf = Wmf[1:].std()
H, n = e13.long_history(fit, fit, 100); H, n = H[evd], n[evd].astype(int)
mask = np.arange(100)[None, :] < n[:, None]
for lam in (1.0 / sd_mf ** 2 / 10, 1.0 / sd_mf ** 2):
    W, b = fit_items(U, lam)
    hv = (Wmf[H] * mask[..., None]).sum(1) / np.maximum(n, 1)[:, None]     # users as the MF pools them
    report(f"(b) MF users, refit items, lam {lam:.0f}", (np.einsum("ed,ekd->ek", hv, W[items6]) + b[items6]).ravel())
hv = (Wmf[H] * mask[..., None]).sum(1) / np.maximum(n, 1)[:, None]
report("MF seed 0 (W, b)", (np.einsum("ed,ekd->ek", hv, Wmf[items6]) + z["b_6"][items6]).ravel())
