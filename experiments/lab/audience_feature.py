# Run from the repository root:
#   .venv/bin/python experiments/lab/audience_feature.py --seq <cache,...> --extra name=<cache> [...]
# Needs data/cache/22_vectors_c7.npz and the caches of the current best feature set (same
# arguments as lab/user_activity.py).
#
# The audience view of lab/audience.py as LightGBM features on top of the current best feature
# set: per candidate, the user vector (hv, mean of known clicks) dotted with the item's AUDIENCE
# (mean taste of its clickers, from their other clicks) and with the audience minus the mean
# taste of the users it was shown to, plus each one's event-relative versions and the audience
# size. All as of each day d (clicks/impressions before d, MF vectors trained on days < d).
# Day-6 screening, alone: MF dot 0.81278, audience 0.81363, contrast 0.81451 (rank corr 0.945).
# Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B.
import argparse, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--extra", action="append", default=[])
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
mf30 = (np.load(str(CACHE / "13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
        + np.load(str(CACHE / "15_mf_c7_46f30c940a6a.npz"))["mf_fit"]) / 2
row_day = np.repeat(fit["day"], 10)
keep = np.flatnonzero(row_day >= 2)
rows, X30, names = e13.feature_matrix(fit, mf30, keep)
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))

def group(caches, prefix):
    S = np.mean([np.load(str(c))["mf_fit"] for c in caches.split(",")], 0)[rows]
    nm = [f"{prefix}_{k}" for k in ("logit", "item_bias", "user_bias", "dot",
                                    "minus_max", "rank", "share", "lse")]
    return np.concatenate([S, e13.relative_mf(S)], 1).astype(np.float32), nm

hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
POP = np.zeros((len(rows), 2), np.float32)
for day in range(2, 7):
    k = fit["clk_ts"] < e13.TRAIN_START + day * e13.DAY
    lp = np.log1p(np.bincount(fit["clk_item"][k], minlength=n_items)).astype(np.float32)
    e = np.flatnonzero(fit["day"] == day)
    hl = hlen[e].astype(np.float32)
    m = np.arange(hist.shape[1])[None, :] < hl[:, None]
    hp = np.where(hl > 0, np.where(m, lp[hist[e]], 0).sum(1) / np.maximum(hl, 1), np.nan)
    cp = lp[fit["item"].reshape(-1, 10)[e]]
    POP[pos[(e[:, None] * 10 + np.arange(10)).ravel()]] = np.stack(
        [np.repeat(hp, 10), cp.ravel() - np.repeat(hp, 10)], 1)
parts, fn = [X30, POP], names + ["hist_pop", "cand_pop_minus_hist"]
g, gn = group(a.seq, "seq"); parts.append(g); fn += gn
for spec_ in a.extra:
    name, caches = spec_.split("=", 1)
    g, gn = group(caches, name); parts.append(g); fn += gn
X = np.concatenate(parts, 1)

# ---- audience features, as of each day
src = (ROOT / "experiments/lab/audience.py").read_text()
ns = {"e13": e13, "np": np, "fit": fit}
exec(src[src.index("def audience"):src.index("for day in")], ns)
audience = ns["audience"]
z = np.load(str(CACHE / "22_vectors_c7.npz"))
AUD = np.zeros((len(rows), 2), np.float32); NA = np.zeros(len(rows), np.float32)
for day in range(2, 7):
    W = z[f"W_{day}"].astype(np.float64); evd = z[f"ev_{day}"]; hv = z[f"hv_{day}"].astype(np.float64)
    A, na, Bm = audience(W, day)
    items = fit["item"].reshape(-1, 10)[evd]
    r = pos[(evd[:, None] * 10 + np.arange(10)).ravel()]
    AUD[r, 0] = np.einsum("ed,ekd->ek", hv, A[items]).ravel()
    AUD[r, 1] = np.einsum("ed,ekd->ek", hv, (A - Bm)[items]).ravel()
    NA[r] = np.log1p(na[items]).ravel()
    print(f"day {day}: audience AUC {auc(AUD[r, 0], y[r]):.5f}, contrast {auc(AUD[r, 1], y[r]):.5f}", flush=True)
sc = AUD.std(0) + 1e-12
AUD = AUD / sc * 3                                 # comparable scale for the softmax share
rel = lambda s: e13.relative_mf(np.stack([s, s, s, s], 1))
FA = np.concatenate([AUD, NA[:, None], rel(AUD[:, 0]), rel(AUD[:, 1])], 1).astype(np.float32)
fa = ["aud_dot", "audc_dot", "aud_log_n"] + [f"aud_{k}" for k in ("minus_max", "rank", "share", "lse")] \
     + [f"audc_{k}" for k in ("minus_max", "rank", "share", "lse")]

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A_, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

import lightgbm as lgb
def run(label, Xm, nm):
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A_], y[A_])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    print(f"  {label:40s} half-B {auc(pb, y[B]):.5f}  half-A {m.best_score['valid_0']['auc']:.5f}  "
          f"({m.best_iteration} trees)", flush=True)
    return m
print("\nLightGBM (lr 0.02):")
run("current best feature set", X, fn)
m = run("+ audience (1 seed of vectors)", np.concatenate([X, FA], 1), fn + fa)
g = sorted(zip(m.feature_importance("gain"), fn + fa), reverse=True)
print("    by gain: " + ", ".join(f"{n_} #{i + 1}" for i, (_, n_) in enumerate(g) if n_ in fa), flush=True)
