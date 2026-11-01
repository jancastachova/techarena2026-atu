# Run from the repository root: .venv/bin/python experiments/lab/replica_user_est.py
# IS THE MEAN OF THE CLICKED VECTORS THE BEST TASTE ESTIMATE? On the 8-dim replica (scale 1.19) with the TRUE item vectors and
# biases, so only the user side is estimated. Day 6, users' events of days 0-5 + warm-up clicks; candidates scored by the
# true taste term theta_hat . phi (+ true bias where noted). Estimators of theta from the same information:
#   mean      unit(mean phi of the known clicks)             (production form)
#   contrast  mean clicked - beta * mean of the 9 losers of each clicked event
#   ML        per-user MAP of the softmax-with-outside-option likelihood of their days 0-5 events (clicked item vs the
#             others vs no click), true biases as offsets, prior theta ~ N(0, s^2 I); outside option per user = a free logit
# and the oracle (true theta). Reported overall and by known clicks.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/replica8_s1.19.npz"))
T = np.load(ROOT / "data/cache/replica8_s1.19_truth.npz")
auc = e13.roc_auc
PHI = T["phi"].astype(np.float64); BI = T["b_item"].astype(np.float64); TH = T["theta"].astype(np.float64)
_, Up = np.linalg.eigh(PHI[1:].T @ PHI[1:]); Up = Up[:, ::-1][:, :8]
Z = PHI @ Up; TH8 = TH @ Up; NI = len(Z)
def unit(A): return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
NU = len(TH)
items = fit["item"].reshape(-1, 10).astype(np.int64); lab = fit["label"].reshape(-1, 10); eu = fit["user"].astype(np.int64)
tr = np.flatnonzero(fit["day"] < 6); d6 = np.flatnonzero(fit["day"] == 6)
warm = fit["clk_ts"] < e13.TRAIN_START
wu, wi = fit["clk_user"][warm].astype(np.int64), fit["clk_item"][warm].astype(np.int64)
clk_ev = tr[lab[tr].max(1) > 0]
npos = np.bincount(eu[clk_ev], minlength=NU) + np.bincount(wu, minlength=NU)
P = np.zeros((NU, 8)); np.add.at(P, eu[clk_ev], Z[items[clk_ev][lab[clk_ev] > 0]]); np.add.at(P, wu, Z[wi])
Lo = np.zeros((NU, 8)); np.add.at(Lo, eu[clk_ev], (Z[items[clk_ev]] * (lab[clk_ev] == 0)[:, :, None]).sum(1) / 9)
nclk = np.bincount(eu[clk_ev], minlength=NU)
Pm = P / np.maximum(npos, 1)[:, None]; Lm = Lo / np.maximum(nclk, 1)[:, None]
import os
if os.environ.get("NO_WARM"):
    # the replica keeps the REAL warm-up clicks and builds the true taste from all real clicks: a mean that includes them
    # partly averages the items that define the truth. Fair version: simulated (days 0-5) clicks only, for every estimator.
    P = np.zeros((NU, 8)); np.add.at(P, eu[clk_ev], Z[items[clk_ev]][lab[clk_ev] > 0])
    npos = nclk.copy(); Pm = P / np.maximum(npos, 1)[:, None]
    print("NO_WARM: warm-up clicks excluded from every estimator; groups by simulated clicks of days 0-5", flush=True)

# ---- ML / MAP per user: softmax over 10 candidates + outside option, batched Newton over all users at once
def map_theta(s2, iters=15, mu=None):
    mu = np.zeros((NU, 8)) if mu is None else mu
    ev = tr; u = eu[ev]; X = Z[items[ev]]; b = BI[items[ev]]; y = lab[ev].astype(np.float64)    # (E,10,8), (E,10)
    th = unit(Pm) * np.linalg.norm(TH8, axis=1).mean() * 0.5; a = np.zeros(NU)                  # start: scaled mean
    for _ in range(iters):
        ut = np.einsum("ekd,ed->ek", X, th[u]) + b
        mx = np.maximum(ut.max(1), a[u]); ex = np.exp(ut - mx[:, None]); eo = np.exp(a[u] - mx)
        den = ex.sum(1) + eo; p = ex / den[:, None]; po = eo / den
        yo = 1 - y.sum(1)
        # gradient wrt theta: sum_k (y_k - p_k) x_k ; wrt a: yo - po
        gth = np.zeros((NU, 8)); np.add.at(gth, u, np.einsum("ek,ekd->ed", y - p, X)); gth -= (th - mu) / s2
        ga = np.bincount(u, yo - po, minlength=NU) - a / 25.0
        # Hessian (theta block): sum E_p[x x^T] - E_p[x] E_p[x]^T  (outside option has x = 0)
        mxp = np.einsum("ek,ekd->ed", p, X)
        Hth = np.zeros((NU, 8, 8))
        np.add.at(Hth, u, np.einsum("ek,ekd,ekf->edf", p, X, X) - mxp[:, :, None] * mxp[:, None, :])
        Hth += np.eye(8) / s2
        Ha = np.bincount(u, po * (1 - po), minlength=NU) + 1 / 25.0
        step = np.linalg.solve(Hth, gth[:, :, None])[:, :, 0]
        nrm = np.linalg.norm(step, axis=1, keepdims=True)
        th = th + step * np.minimum(1.0, 0.5 / np.maximum(nrm, 1e-12)); a = a + np.clip(ga / Ha, -1, 1)
        ll = (np.log(np.maximum((p * y).sum(1) + po * yo, 1e-300))).mean()
        if _ == iters - 1: print(f"    iter {_}: mean log-lik {ll:.4f}  |theta| {np.linalg.norm(th, axis=1).mean():.3f}  "
              f"cos(theta, true) {np.mean(np.einsum('nd,nd->n', unit(th), unit(TH8))[npos > 0]):.3f}", flush=True)
    return th
u6 = eu[d6]; it6 = items[d6]; y6 = lab[d6].ravel()
keep = npos[u6] > 0; u6, it6, y6 = u6[keep], it6[keep], lab[d6][keep].ravel()
kn = np.repeat(npos[u6], 10)
GR = [("1-5", (kn >= 1) & (kn <= 5)), ("6-20", (kn >= 6) & (kn <= 20)), ("21+", kn >= 21)]
def rep(name, th, bias=False):
    s = np.einsum("ed,ekd->ek", th[u6], Z[it6]) + (BI[it6] if bias else 0)
    s = s.ravel()
    print(f"{name:34s} AUC {auc(s, y6):.5f} | " + "  ".join(f"{g} {auc(s[m], y6[m]):.4f}" for g, m in GR), flush=True)
print(f"day-6 events {len(u6)}; true taste norm {np.linalg.norm(TH8[u6], axis=1).mean():.2f}", flush=True)
rep("oracle theta", TH8)
rep("mean (unit)", unit(Pm))
for beta in (0.3, 1.0):
    rep(f"contrast beta {beta}", unit(Pm - beta * Lm))
# priors matched to the truth: per-direction variance of TH8 ~0.37; and centred on the scaled click mean (Bayes-style
# combination of the two estimators; scale c fitted as the mean projection of the true taste on the unit click mean)
c = np.mean(np.einsum("nd,nd->n", TH8, unit(Pm))[npos > 0])
for s2, mu, tag in [(0.37, None, "0"), (0.1, None, "0"), (0.37, c * unit(Pm), "c x mean"), (0.1, c * unit(Pm), "c x mean"),
                    (0.03, c * unit(Pm), "c x mean")]:
    th = map_theta(s2, mu=mu)
    rep(f"MAP prior var {s2}, mean {tag}", th)
    rep(f"MAP prior var {s2}, mean {tag}, unit", unit(th))
