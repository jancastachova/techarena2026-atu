# Run from the repository root: .venv/bin/python experiments/lab/ease_poptaste.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Two cheap LightGBM feature ideas on top of experiment 13 (official split, fit period only):
#  - EASE (closed-form item-item model) on the N most-clicked items, as of each fit day
#  - popularity taste: mean log-popularity of the user's known clicks, and candidate minus that
# Train fit days 2-5; day 6 split by event: early stopping on half A, AUC reported on half B.
import importlib.util, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, n_users, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
N_TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
LAMBDAS = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [200.0]
auc = e13.roc_auc

hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)            # known clicks strictly before each event
row_day = np.repeat(fit["day"], 10)
cu, ci, ct = fit["clk_user"], fit["clk_item"], fit["clk_ts"]

def ease_scores(day, lam):
    """EASE on clicks before `day` (top N_TOP items); scores for the rows of fit day `day`."""
    keep = ct < e13.TRAIN_START + day * e13.DAY
    cnt = np.bincount(ci[keep], minlength=n_items)
    top = np.argsort(-cnt)[:N_TOP]
    top = top[cnt[top] > 0]
    idx = np.full(n_items, -1); idx[top] = np.arange(len(top))
    k = keep & (idx[ci] >= 0)
    pairs = np.unique(np.stack([cu[k], idx[ci[k]]], 1), axis=0)
    X = csr_matrix((np.ones(len(pairs), np.float32), (pairs[:, 0], pairs[:, 1])), shape=(n_users, len(top)))
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = (P / -np.diag(P)).astype(np.float32)
    B[np.diag_indices_from(B)] = 0
    del G, P
    ev = np.flatnonzero(fit["day"] == day)
    out = np.full((len(ev), 10), np.nan, np.float32)
    cand = idx[fit["item"].reshape(-1, 10)[ev]]                    # -1 = not in the EASE item set
    h = idx[hist[ev]]
    h[np.arange(h.shape[1])[None, :] >= hlen[ev][:, None]] = -1    # padding
    for s in range(0, len(ev), 2000):
        hh, cc = h[s:s + 2000], cand[s:s + 2000]
        g = B[np.maximum(hh, 0)[:, :, None], np.maximum(cc, 0)[:, None, :]]
        g = np.where((hh >= 0)[:, :, None], g, 0).sum(1)
        out[s:s + 2000] = np.where(cc >= 0, g, np.nan)
    return ev, out

def pop_taste(day):
    keep = ct < e13.TRAIN_START + day * e13.DAY
    lp = np.log1p(np.bincount(ci[keep], minlength=n_items)).astype(np.float32)
    ev = np.flatnonzero(fit["day"] == day)
    hl = hlen[ev].astype(np.float32)
    m = np.arange(hist.shape[1])[None, :] < hl[:, None]
    hp = np.where(m, lp[hist[ev]], 0).sum(1) / np.maximum(hl, 1)
    hp = np.where(hl > 0, hp, np.nan)
    cp = lp[fit["item"].reshape(-1, 10)[ev]]
    return ev, np.stack([np.repeat(hp, 10), cp.ravel() - np.repeat(hp, 10)], 1)

rows, Xb, names = e13.feature_matrix(fit, mf, np.flatnonzero(row_day >= 2))
assert np.array_equal(rows, np.flatnonzero(row_day >= 2))
y, d, ev_of = fit["label"][rows], row_day[rows], rows // 10
pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))

POP = np.full((len(rows), 2), np.nan, np.float32)
for day in range(2, 7):
    ev, f = pop_taste(day)
    POP[pos[(ev[:, None] * 10 + np.arange(10)).ravel()]] = f

rng = np.random.default_rng(0)
ev6 = np.unique(ev_of[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B_ = d <= 5, (d == 6) & a_set[ev_of], (d == 6) & ~a_set[ev_of]

import lightgbm as lgb
def run(label, X, nm):
    m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=nm), 3000,
                  valid_sets=[lgb.Dataset(X[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"  {label:34s} A {m.best_score['valid_0']['auc']:.5f}  B "
          f"{auc(m.predict(X[B_], num_iteration=m.best_iteration), y[B_]):.5f}  ({m.best_iteration} trees)", flush=True)

print(f"LightGBM, fit days 2-5 -> day 6 (half A: early stopping, half B: report)", flush=True)
run("13 as is", Xb, names)
run("+ popularity taste", np.concatenate([Xb, POP], 1), names + ["hist_pop", "cand_pop_minus_hist"])
for lam in LAMBDAS:
    EASE = np.full((len(rows), 1), np.nan, np.float32)
    t = time.time()
    for day in range(2, 7):
        ev, s = ease_scores(day, lam)
        EASE[pos[(ev[:, None] * 10 + np.arange(10)).ravel()], 0] = s.ravel()
    ok = ~np.isnan(EASE[:, 0]) & (d == 6)
    print(f"EASE top {N_TOP}, lambda {lam:g} ({time.time()-t:.0f}s): covers {np.mean(~np.isnan(EASE[d == 6, 0])):.1%} "
          f"of day-6 rows, AUC there {auc(EASE[ok, 0], y[ok]):.5f} (MF on same rows {auc(mf[rows][ok, 0], y[ok]):.5f})", flush=True)
    run(f"+ EASE (lambda {lam:g})", np.concatenate([Xb, EASE], 1), names + ["ease"])
    run(f"+ EASE + popularity taste", np.concatenate([Xb, EASE, POP], 1), names + ["ease", "hist_pop", "cand_pop_minus_hist"])
