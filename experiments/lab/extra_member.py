# Run from the repository root:
#   .venv/bin/python experiments/lab/extra_member.py --seq <cache,...> --extra <name>=<cache,...>
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and cached arrays in data/cache.
#
# Does a further alternative MF (e.g. experiment 21's deep interaction, which scores about as
# well alone as the standalone sequence model) add information ON TOP of the sequence features
# that gave the current best (24_ensemble_seq1.csv, public 0.81853)? The alternative model's 4
# scores plus their event-relative versions are added as LightGBM features next to MF30 +
# popularity taste + sequence features; its rank correlation with the MF and with the sequence
# model is printed first (diversity is what makes such a member useful).
# Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B.
import argparse, importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True, help="comma-separated 18 --pure-seq caches (averaged)")
ap.add_argument("--extra", action="append", default=[], help="name=cache[,cache...] (averaged)")
ap.add_argument("--base-extra", action="append", default=[],
                help="name=cache[,...]: groups ALWAYS in the base (e.g. the IDF groups of the current best)")
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

def group(caches, prefix):
    S = np.mean([np.load(str(c))["mf_fit"] for c in caches.split(",")], 0)[rows]
    if np.isnan(S).any():
        raise SystemExit(f"{prefix}: scores missing for some rows")
    nm = [f"{prefix}_{k}" for k in ("logit", "item_bias", "user_bias", "dot",
                                    "minus_max", "rank", "share", "lse")]
    return np.concatenate([S, e13.relative_mf(S)], 1).astype(np.float32), nm, S[:, 0]

hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
POP = np.zeros((len(rows), 2), np.float32)
pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))
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

SEQ, snames, s0 = group(a.seq, "seq")
base_parts, base_names = [X30, POP, SEQ], names + ["hist_pop", "cand_pop_minus_hist"] + snames
for spec_ in a.base_extra:
    name, caches = spec_.split("=", 1)
    g, gn, _ = group(caches, name)
    base_parts.append(g); base_names += gn
base_X = np.concatenate(base_parts, 1)
print(f"base: MF30 + popularity + sequence + {[s_.split('=')[0] for s_ in a.base_extra]}")
d6 = d == 6
mf0 = X30[:, names.index("mf_logit")]
corr = lambda u, v: np.corrcoef(rankdata(u[d6]), rankdata(v[d6]))[0, 1]
print(f"sequence ({len(a.seq.split(','))} cache(s)): day-6 AUC {auc(s0[d6], y[d6]):.5f}, "
      f"rank corr with MF30 {corr(s0, mf0):.4f}")
extras = {}
for spec_ in a.extra:
    name, caches = spec_.split("=", 1)
    Xe, en, e0 = group(caches, name)
    extras[name] = (Xe, en)
    print(f"{name} ({len(caches.split(','))} cache(s)): day-6 AUC {auc(e0[d6], y[d6]):.5f}, "
          f"rank corr with MF30 {corr(e0, mf0):.4f}, with sequence {corr(e0, s0):.4f}", flush=True)

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

import lightgbm as lgb
def run(label, Xm, nm):
    p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
    m = lgb.train(p, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 8000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
    pb = m.predict(Xm[B], num_iteration=m.best_iteration)
    print(f"  {label:44s} half-B {auc(pb, y[B]):.5f}  ({m.best_iteration} trees)", flush=True)
    return m
print("\nLightGBM (lr 0.02) on MF30 + popularity taste + sequence, and with each extra model:")
run("base (= the best member)", base_X, base_names)
for name, (Xe, en) in extras.items():
    m = run(f"+ {name}", np.concatenate([base_X, Xe], 1), base_names + en)
    g = sorted(zip(m.feature_importance("gain"), base_names + en), reverse=True)
    top = [f"{n} #{i + 1}" for i, (_, n) in enumerate(g) if n in en][:3]
    print(f"    best-ranked {name} features by gain: " + ", ".join(top))
