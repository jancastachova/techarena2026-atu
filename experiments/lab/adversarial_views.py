# Run from the repository root: bash -c 'source experiments/lab/bundle_args.sh; .venv/bin/python experiments/lab/adversarial_views.py "${SET38[@]}"'
# ADVERSARIAL CHECK OF THE EXTRA VIEWS (2026-10-30). Since 36 the lab gains of new views transfer to the leaderboard at a
# fraction of their size (36: +0.00059 eval-weighted -> +0.00006 public; 38 alone below 37c). lab/adversarial.py checked
# only 13's features. Here every column of every SET38 view (24_ensemble --extra caches: mf_fit / mf_test, 1 row per
# candidate) is compared between each user's FIRST day-6 event (reweighted to the evaluation's known-clicks mix) and the
# evaluation rows: per-column AUC (0.5 = same distribution) and medians. Columns far from 0.5 without a legitimate reason
# (longer gaps, lighter histories, stats frozen at the cutoff) point at a feature computed differently for the test rows.
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
d6 = np.flatnonzero(fit["day"] == 6)
_, first = np.unique(fit["user"][d6], return_index=True); ev6 = d6[first]
rows6 = (ev6[:, None] * 10 + np.arange(10)).ravel()
_, hl6 = e13.long_history({"user": fit["user"][ev6], "ts": fit["ts"][ev6]}, fit, 100)
_, hlT = e13.long_history(test, fit, 100)
B = [0, 5, 20, 50, 100]
def bk(h): return np.searchsorted(B, np.asarray(h, int))
sT = np.bincount(bk(hlT), minlength=5) / len(hlT); s6 = np.bincount(bk(hl6), minlength=5) / len(hl6)
w6 = np.repeat((sT / np.maximum(s6, 1e-9))[bk(hl6)], 10)
def wauc(a, b, wa):                      # weighted AUC of "b (evaluation) above a (day 6)"
    v = np.r_[a, b]; y = np.r_[np.zeros(len(a)), np.ones(len(b))]; w = np.r_[wa, np.ones(len(b))]
    o = np.argsort(v, kind="mergesort"); v, y, w = v[o], y[o], w[o]
    _, start, cnt = np.unique(v, return_index=True, return_counts=True)
    neg = np.add.reduceat(w * (y == 0), start); pos = np.add.reduceat(w * (y == 1), start)
    cneg = np.cumsum(neg) - neg
    return float((pos * (cneg + neg / 2)).sum() / (pos.sum() * neg.sum()))
print(f"day-6 first events {len(ev6)}, evaluation events {len(hlT)}", flush=True)
args = sys.argv[1:]
specs = [a for a in args if "=" in a and not a.startswith("--")]
out = []
for sp in specs:
    name, _, files = sp.partition("=")
    Zs = [np.load(f) for f in files.split(",")]
    F = np.mean([z["mf_fit"] for z in Zs], 0); T = np.mean([z["mf_test"] for z in Zs], 0)
    if F.ndim == 1: F, T = F[:, None], T[:, None]
    for c in range(F.shape[1]):
        a, b = F[rows6, c].astype(np.float64), T[:, c].astype(np.float64)
        if np.all(a == a[0]) and np.all(b == b[0]): continue
        u = wauc(a, b, w6)
        out.append((abs(u - 0.5), name, c, u, np.median(a), np.median(b)))
        print(f"{name:12s} col {c}: AUC eval-vs-day6 {u:.3f}  median day6 {np.median(a):+.4f} eval {np.median(b):+.4f}", flush=True)
print("\nmost separating columns:")
for d, name, c, u, ma, mb in sorted(out, reverse=True)[:15]:
    print(f"  {name:12s} col {c}: {u:.3f}  ({ma:+.4f} -> {mb:+.4f})")
