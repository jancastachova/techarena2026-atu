# Run from the repository root: .venv/bin/python experiments/lab/user_sharpness.py
# Is TASTE SHARPNESS a user trait? LightGBM knows how often a user clicks, not how taste-driven their choices are.
# Per clicked event (days 2-6), the clicked item's within-event percentile by MF30's time-forward score (the MF of
# that event never saw its day) and the clicked item's softmax share. Split-half reliability across each user's
# clicked events (odd vs even) for users with >= 10 clicked events, against a shuffle null.
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.mean([np.load(ROOT / "data/cache" / f)["mf_fit"][:, 0] for f in
              ("13_mf_c7_bb1fcb0a9c89.npz", "15_mf_c7_46f30c940a6a.npz")], 0).reshape(-1, 10)
Y = fit["label"].reshape(-1, 10)
ev = np.flatnonzero((fit["day"] >= 2) & (Y.max(1) > 0))
S = mf[ev]; c = Y[ev].argmax(1)
pct = (rankdata(S, axis=1)[np.arange(len(ev)), c] - 1) / 9.0              # 1 = clicked the top-scored candidate
sh = np.exp(S - S.max(1, keepdims=True)); sh = (sh / sh.sum(1, keepdims=True))[np.arange(len(ev)), c]
u = fit["user"][ev]; o = np.lexsort((fit["ts"][ev], u)); u, pct, sh = u[o], pct[o], sh[o]
r = np.arange(len(u)) - np.searchsorted(u, u, "left")
cnt = np.bincount(u); ok_u = np.flatnonzero(cnt >= 10)
def halves(x, uu, rr):
    odd = rr % 2 == 1
    a = np.bincount(uu[odd], x[odd], minlength=len(cnt)) / np.maximum(np.bincount(uu[odd], minlength=len(cnt)), 1)
    b = np.bincount(uu[~odd], x[~odd], minlength=len(cnt)) / np.maximum(np.bincount(uu[~odd], minlength=len(cnt)), 1)
    return np.corrcoef(a[ok_u], b[ok_u])[0, 1]
rng = np.random.default_rng(0)
for name, x in (("clicked-item percentile", pct), ("clicked-item MF share", sh)):
    null = [halves(x[rng.permutation(len(x))], u, r) for _ in range(5)]
    print(f"{name:26s} mean {x.mean():.3f}  split-half reliability over {len(ok_u)} users {halves(x, u, r):+.3f}  "
          f"(shuffled across users {np.mean(null):+.3f})")
