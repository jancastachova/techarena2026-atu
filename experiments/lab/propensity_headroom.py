# Run from the repository root: .venv/bin/python experiments/lab/propensity_headroom.py
# DIAGNOSTIC ONLY (uses later events' labels; never a feature). How much global AUC would a better per-user click-propensity
# estimate buy? Day-6 half B, lean LightGBM predictions p. Event level E = sum of p over the event. A logistic model on half A
# re-levels each event: logit(E') = a + b logit(E) + c x (user propensity estimate) and rows are rescaled p' = p E'/E.
# Estimates: "past" = shrunk click share of the user's events before the event (what the model has); "loo" = shrunk click
# share of ALL the user's other events in days 0-6 (past and FUTURE: an oracle-ish propensity). Gain of loo over past = the
# headroom of the propensity side, by known-clicks group.
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location("e13", "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, *_ = e13.load_base("data/cache/13_base_c7_9c47a117c762.npz")
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
z = np.load("data/cache/lab_preds_lean0_d6.npz"); rB = rows[z["rows_B"]]; pB = z["base (= the best member)|B"]
# half A rows: day-6 rows not in B
d6 = np.flatnonzero(np.repeat(fit["day"], 10) == 6); inB = np.zeros(len(fit["label"]), bool); inB[rB] = True
rA = d6[~inB[d6]]
ev_click = fit["label"].reshape(-1, 10).max(1).astype(float)
u = fit["user"].astype(np.int64); ts = fit["ts"]
NU = u.max() + 1
tot_c = np.bincount(u, ev_click, NU); tot_n = np.bincount(u, minlength=NU).astype(float)
o = np.lexsort((ts, u)); cum_c = np.zeros(len(u)); cum_n = np.zeros(len(u))
cc = np.cumsum(ev_click[o]); first = np.r_[True, u[o][1:] != u[o][:-1]]
start = np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
cum_c[o] = cc - ev_click[o] - np.r_[0, cc][start]; cum_n[o] = np.arange(len(o)) - start
g, K = ev_click.mean(), 5.0
past = (cum_c + K * g) / (cum_n + K)
loo = (tot_c[u] - ev_click + K * g) / (tot_n[u] - 1 + K)
_, hl = e13.long_history(fit, fit, 100)
def lg(x): x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))
H, _ = None, None
def fitlr(X, y, it=40):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        q = 1 / (1 + np.exp(-X @ w)); w += np.linalg.solve((X * (q * (1 - q))[:, None]).T @ X + 1e-6 * np.eye(len(w)), X.T @ (y - q))
    return w
def relevel(rows_fit, p_fit, rows_ap, p_ap, est):
    ef, ea = rows_fit[::10] // 10, rows_ap[::10] // 10
    Ef, Ea = np.clip(p_fit.reshape(-1, 10).sum(1), 1e-4, 0.9999), np.clip(p_ap.reshape(-1, 10).sum(1), 1e-4, 0.9999)
    Xf = np.c_[np.ones(len(ef)), lg(Ef), lg(est[ef])]; Xa = np.c_[np.ones(len(ea)), lg(Ea), lg(est[ea])]
    w = fitlr(Xf, ev_click[ef]); Ea2 = 1 / (1 + np.exp(-Xa @ w))
    return (p_ap.reshape(-1, 10) * (Ea2 / Ea)[:, None]).ravel(), w
# half-A predictions are not saved; use half B split in two by event for fit / apply
evB = rB[::10] // 10; rng = np.random.default_rng(1); h = rng.random(len(evB)) < 0.5
mf = np.repeat(h, 10); ma = ~mf
yB = fit["label"][rB].astype(float)
base = e13.roc_auc(pB[ma], yB[ma])
print(f"apply half of half B ({ma.sum() // 10} events): model as is {base:.5f}")
for name, est in (("past", past), ("loo (oracle-ish)", loo)):
    p2, w = relevel(rB[mf], pB[mf], rB[ma], pB[ma], est)
    print(f"  re-levelled with {name:18s} {e13.roc_auc(p2, yB[ma]):.5f}  (coef on estimate {w[2]:+.3f})")
    hh = hl[rB[ma] // 10]
    print("     by known clicks 1-5 / 6-20 / 21+: " + " / ".join(
        f"{e13.roc_auc(p2[(hh >= lo) & (hh <= hi)], yB[ma][(hh >= lo) & (hh <= hi)]):.4f} vs {e13.roc_auc(pB[ma][(hh >= lo) & (hh <= hi)], yB[ma][(hh >= lo) & (hh <= hi)]):.4f}"
        for lo, hi in ((1, 5), (6, 20), (21, 100))))
