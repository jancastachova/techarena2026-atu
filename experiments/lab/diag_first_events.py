# Run from the repository root: .venv/bin/python experiments/lab/diag_first_events.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Where does experiment 13 lose AUC on first events? Validation cutoff 5 (test = each user's
# first event at/after day 5), model = the same LightGBM procedure as the submission.
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c5_a2ecdd59dd03.npz"))
z = np.load(str(ROOT / "data/cache/13_mf_c5_2889ef80775f.npz"))
model = e13.lgbm_procedure(fit, z["mf_fit"], 5)
rows, X, _ = e13.feature_matrix(test, z["mf_test"], np.arange(len(test["item"])))
p, y = model.predict(X), test["label"].astype(np.int64)
auc = e13.roc_auc
E = len(test["user"])
P, Y = p.reshape(E, 10), y.reshape(E, 10)
print(f"\nfirst events: {E:,}, rows {len(y):,}, click rate {y.mean():.4f}, events with click {Y.any(1).mean():.3f}")
print(f"global AUC {auc(p, y):.5f}   MF alone {auc(z['mf_test'][:, 0], y):.5f}")

# 1) event level vs within event
clicked = Y.any(1)
print(f"event-level AUC (sum of p vs 'event has a click') {auc(P.sum(1), clicked.astype(int)):.5f}")
r = (P[clicked] > P[clicked][Y[clicked] == 1][:, None]).sum(1)          # candidates ranked above the click
print(f"within clicked events: click ranked 1st {np.mean(r == 0):.3f}, mean rank {r.mean() + 1:.2f}, "
      f"within-event AUC {1 - r.mean() / 9:.4f}")
Pw = P.copy()                                                           # perfect ranking inside events
for i in np.flatnonzero(clicked):
    s = np.sort(P[i]); j = int(np.flatnonzero(Y[i])[0]); others = [k for k in range(10) if k != j]
    Pw[i, j] = s[-1]; Pw[i, others] = s[:-1][np.argsort(np.argsort(P[i, others]))]
print(f"counterfactual, perfect within-event ranking: global AUC {auc(Pw.ravel(), y):.5f}")
Pe = np.where(clicked[:, None], P, P * 0.5)                             # halve scores of no-click events
print(f"counterfactual, no-click events scored half as high: global AUC {auc(Pe.ravel(), y):.5f}")

# 2) segments
n_clk = np.bincount(fit["clk_user"], minlength=int(test["user"].max()) + 1)   # user's known pre-cutoff clicks
uc = np.repeat(n_clk[test["user"]], 10)
impr = np.bincount(fit["item"], minlength=n_items)[test["item"]]
clk_i = np.bincount(fit["item"], weights=fit["label"], minlength=n_items)[test["item"]]
hrs = np.repeat((test["ts"] - (e13.TRAIN_START + 5 * e13.DAY)) / 3600, 10)
def show(title, segs):
    print(f"\n{title}")
    print(f"  {'segment':24s} {'rows':>7s} {'clk rate':>8s} {'AUC':>8s}")
    for name, m in segs:
        if m.sum() and 0 < y[m].mean() < 1:
            print(f"  {name:24s} {m.mean():7.1%} {y[m].mean():8.4f} {auc(p[m], y[m]):8.5f}")
show("user's known clicks before the cutoff", [
    ("0", uc == 0), ("1-5", (uc >= 1) & (uc <= 5)), ("6-20", (uc >= 6) & (uc <= 20)),
    ("21-100", (uc > 20) & (uc <= 100)), (">100", uc > 100)])
show("candidate impressions in the fit period", [
    ("never shown", impr == 0), ("1-100", (impr >= 1) & (impr <= 100)), ("101-1000", (impr > 100) & (impr <= 1000)),
    ("1001-10000", (impr > 1000) & (impr <= 10000)), (">10000", impr > 10000)])
show("candidate clicks in the fit period", [
    ("0", clk_i == 0), ("1-10", (clk_i >= 1) & (clk_i <= 10)), ("11-100", (clk_i > 10) & (clk_i <= 100)), (">100", clk_i > 100)])
show("hours after the cutoff", [
    ("0-6", hrs < 6), ("6-24", (hrs >= 6) & (hrs < 24)), ("24-48", hrs >= 24)])
