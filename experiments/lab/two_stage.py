# Run from the repository root: .venv/bin/python experiments/lab/two_stage.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
#
# Why: the global AUC splits into "which EVENTS get a click" and "which candidate inside an
# event". The first-event diagnosis (lab/diag_first_events.py) found the event part weak
# (AUC 0.67 from the sum of the row scores) while scoring no-click events half as high would
# give +0.038 global AUC. The row model is trained per row, so nothing optimises the event part
# directly. Here a SECOND LightGBM is trained on one row per event (label = "any click"), and
# its prediction is combined with the row model two ways.
# Fit days 2-5; day 6 split by event: half A for early stopping and the combination weight,
# AUC reported on half B (row model alone 0.83571).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
auc = e13.roc_auc
row_day = np.repeat(fit["day"], 10)
rows, X, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
y = fit["label"][rows]
ev_day = fit["day"][fit["day"] >= 2]
n_ev = len(X) // 10
assert len(ev_day) == n_ev
col = {n: i for i, n in enumerate(names)}

# ---- event-level design matrix: the event features once, plus aggregates over the 10 candidates
G = lambda name: X[:, col[name]].reshape(n_ev, 10)
parts, enames = [], []
for n in e13.EVENT_FEATURES + ["mf_user_bias", "mf_event_lse"]:
    parts.append(G(n)[:, :1]); enames.append(n)
for n in ["mf_logit", "mf_hist_dot", "mf_item_bias", "item_ctr_logit", "sim_sum", "rep_any"]:
    g = G(n)
    parts += [g.max(1, keepdims=True), g.mean(1, keepdims=True), g.std(1, keepdims=True)]
    enames += [f"{n}_max", f"{n}_mean", f"{n}_std"]
Xe = np.concatenate(parts, 1).astype(np.float32)
ye = y.reshape(n_ev, 10).any(1).astype(np.int8)

rng = np.random.default_rng(0)
ev6 = np.flatnonzero(ev_day == 6)
a_ev = np.zeros(n_ev, bool); a_ev[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr_e, A_e, B_e = ev_day <= 5, (ev_day == 6) & a_ev, (ev_day == 6) & ~a_ev
rowmask = lambda m: np.repeat(m, 10)
tr, A, B = rowmask(tr_e), rowmask(A_e), rowmask(B_e)

import lightgbm as lgb
def train(Xm, ym, trm, am, rounds=3000):
    m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(Xm[trm], ym[trm]), rounds,
                  valid_sets=[lgb.Dataset(Xm[am], ym[am])], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.best_iteration

mr, it_r = train(X, y, tr, A)
pA, pB = mr.predict(X[A], num_iteration=it_r), mr.predict(X[B], num_iteration=it_r)
row_B = auc(pB, y[B])
print(f"row model (13)           half-B {row_B:.5f}  ({it_r} trees)", flush=True)

me, it_e = train(Xe, ye, tr_e, A_e)
eA, eB = me.predict(Xe[A_e], num_iteration=it_e), me.predict(Xe[B_e], num_iteration=it_e)
sumA, sumB = pA.reshape(-1, 10).sum(1), pB.reshape(-1, 10).sum(1)
print(f"\n'event gets a click' AUC on half B: event model {auc(eB, ye[B_e]):.5f}  "
      f"vs sum of row scores {auc(sumB, ye[B_e]):.5f}  ({it_e} trees)", flush=True)

lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
def combo_add(p, e, beta):                       # row log-odds + beta * event log-odds
    return lg(p) + beta * np.repeat(lg(e), 10)
def combo_share(p, e, beta):                     # within-event share x event probability
    g = p.reshape(-1, 10)
    return (g / g.sum(1, keepdims=True) * (np.repeat(e, 10).reshape(-1, 10) ** beta)).ravel()

print()
for name, f, grid in [("row + beta*event (log-odds)", combo_add, [0.1, 0.25, 0.5, 0.75, 1.0]),
                      ("within-event share x event^beta", combo_share, [0.5, 0.75, 1.0, 1.25])]:
    scores = [(auc(f(pA, eA, b), y[A]), b) for b in grid]
    best_a, b = max(scores)
    line = "  ".join(f"b={b_:g}: A {a:.5f} B {auc(f(pB, eB, b_), y[B]):.5f}" for a, b_ in scores)
    print(f"{name}\n  {line}")
    print(f"  chosen on A: beta {b:g} -> half B {auc(f(pB, eB, b), y[B]):.5f}  "
          f"gain {auc(f(pB, eB, b), y[B]) - row_B:+.5f}\n", flush=True)
