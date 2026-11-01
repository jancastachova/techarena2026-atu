# Run from the repository root: .venv/bin/python experiments/lab/user_repeat_rate.py
# DO USERS DIFFER IN HOW MUCH THEY RE-CLICK? Repeat candidates (already in the user's known clicks) are 2.6% of rows but
# click at ~0.29, so they carry ~12% of all clicks. The base features describe the CANDIDATE's repeat status (rep_any,
# rep_count, rep_recency); nothing describes the USER's re-click habit. Per user, as of day 6 (events before day 6 only):
#   hist_rep   1 - distinct / total over the known clicks (up to 100)
#   rep_rate   among the user's earlier events that showed >= 1 repeat candidate: share in which a repeat was clicked
#              (shrunk to the global rate, k = 3), and the count of such events
# Tested on day-6 half-B repeat rows against the SET36 binary LightGBM's predictions (lab/global_rank2.py preds): logistic
# recalibration logit(p) -> logit(p) + b * stat fitted on one half of the users' events, AUC on the other half (repeat rows
# and all rows), and the stat's AUC within model-prediction deciles.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
auc = e13.roc_auc
C = ROOT / "data/cache"
fit, _, _, _, _ = e13.load_base(str(C / "13_base_c7_9c47a117c762.npz"))
items = fit["item"].reshape(-1, 10).astype(np.int64); lab = fit["label"].reshape(-1, 10); eu = fit["user"].astype(np.int64)
NU = int(eu.max()) + 1
rep = fit["cand"][:, e13.CAND_FEATURES.index("rep_any")].reshape(-1, 10) > 0
H, n = e13.long_history(fit, fit, 100)
pre = np.flatnonzero((fit["day"] < 6) & rep.any(1))
hit = (lab[pre] * rep[pre]).sum(1) > 0
cnt = np.bincount(eu[pre], minlength=NU)
g = hit.mean()
rr = (np.bincount(eu[pre], hit, NU) + 3 * g) / (cnt + 3)
print(f"events before day 6 with a repeat candidate: {len(pre)}, repeat clicked in {g:.3f}")
# split-half reliability of the per-user repeat rate (users with >= 10 such events)
par = np.random.default_rng(0).random(len(pre)) < 0.5
ra = np.bincount(eu[pre][par], hit[par], NU) / np.maximum(np.bincount(eu[pre][par], minlength=NU), 1)
rb = np.bincount(eu[pre][~par], hit[~par], NU) / np.maximum(np.bincount(eu[pre][~par], minlength=NU), 1)
m10 = cnt >= 10
print(f"users with >= 10 such events: {m10.sum()}, split-half corr of their repeat-click rate {np.corrcoef(ra[m10], rb[m10])[0, 1]:.3f}")
# day-6 rows and the SET36 binary predictions on half B
z = np.load(C / "lab_preds_global_rank_set36.npz")
rows_all = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
rows = rows_all[z["rows_B"]]; p = z["binary|B"]
ev = rows // 10; y = fit["label"][rows].astype(float); rp = rep.ravel()[rows]
hl = n[ev].astype(int)
Hh = H[ev].astype(np.int64); msk = np.arange(100)[None] < hl[:, None]
distinct = np.array([len(set(r[m_])) for r, m_ in zip(Hh[::10], msk[::10])]).repeat(10)
hist_rep = 1 - distinct / np.maximum(hl, 1)
stats = {"hist_rep": hist_rep, "rep_rate": rr[eu[ev]], "log rep events": np.log1p(cnt[eu[ev]])}
lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
print(f"half-B repeat rows {rp.sum()} (click rate {y[rp].mean():.3f}); model AUC on repeat rows {auc(p[rp], y[rp]):.4f}, all {auc(p, y):.5f}")
def logit_fit(X, yy, it=30):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        q = 1 / (1 + np.exp(-X @ w)); gr = X.T @ (yy - q); Hs = (X * (q * (1 - q))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        w += np.linalg.solve(Hs, gr)
    return w
uev = np.unique(ev); half = np.zeros(ev.max() + 1, bool); half[np.random.default_rng(1).choice(uev, len(uev) // 2, replace=False)] = True
h1 = half[ev]
for name, s_ in stats.items():
    dec = np.digitize(p[rp], np.quantile(p[rp], np.linspace(0.1, 0.9, 9)))
    within = np.nanmean([auc(s_[rp][dec == k], y[rp][dec == k]) for k in range(10) if 0 < y[rp][dec == k].sum() < (dec == k).sum()])
    # recalibrate on repeat rows only: logit + b * standardised stat (interaction with rep), fitted on half 1
    zs = (s_ - s_[rp].mean()) / (s_[rp].std() + 1e-12) * rp
    X = np.c_[np.ones(len(p)), lg, zs]
    w = logit_fit(X[h1], y[h1])
    q = X[~h1] @ w; q0 = X[~h1][:, :2] @ logit_fit(X[h1][:, :2], y[h1])
    print(f"{name:15s} AUC alone on repeat rows {auc(s_[rp], y[rp]):.4f}, within model deciles {within:.4f} | coef {w[2]:+.3f} | "
          f"held-out all rows {auc(q0, y[~h1]):.5f} -> {auc(q, y[~h1]):.5f}, repeat rows {auc(q0[rp[~h1]], y[~h1][rp[~h1]]):.4f} -> "
          f"{auc(q[rp[~h1]], y[~h1][rp[~h1]]):.4f}", flush=True)
