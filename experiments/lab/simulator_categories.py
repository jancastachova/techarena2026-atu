# Run from the repository root: .venv/bin/python experiments/lab/simulator_categories.py
# Needs data/cache/22_vectors_c7.npz (experiments/22_product_dims/product_dims.py --save-vectors)
# and the base cache of experiment 13.
#
# Reverse-engineering the simulator: are there hidden CATEGORIES? Synthetic recommendation data
# is often generated with a discrete genre structure (each item has a category, each user
# prefers a few). If so, the 32-dim MF only approximates a much simpler truth, and a model that
# recovers the categories would be far sharper -- especially for light users.
# Test: k-means on the MF's item vectors (trained on all fit days), then measure how strongly
# each user's clicks CONCENTRATE in their top cluster, against a shuffled baseline with the same
# cluster sizes. Also: do the candidates shown to a user come from the user's own clusters?
from pathlib import Path
import numpy as np
from scipy.cluster.vq import kmeans2
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
import importlib.util
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
W = np.load(str(CACHE / "22_vectors_c7.npz"))["W_test"].astype(np.float64)   # item vectors, all fit days
clk_n = np.bincount(fit["clk_item"], minlength=n_items)
ok = clk_n >= 5                                       # items with enough clicks for a stable vector
ok[0] = False
V = W[ok] / np.linalg.norm(W[ok], axis=1, keepdims=True)
print(f"items with >= 5 clicks: {ok.sum():,} of {n_items - 1:,}")

# users' clicks (click log), restricted to clustered items
cu, ci = fit["clk_user"], fit["clk_item"]
keep = ok[ci]
cu, ci = cu[keep], ci[keep]
idx = np.full(n_items, -1); idx[np.flatnonzero(ok)] = np.arange(ok.sum())
n_per_user = np.bincount(cu)
heavy = np.flatnonzero(n_per_user >= 15)
print(f"users with >= 15 clicks on clustered items: {len(heavy):,}\n")
rng = np.random.default_rng(0)

def concentration(labels, k):
    """Mean share of a heavy user's clicks in their single most-clicked cluster."""
    lab = labels[idx[ci]]
    key = cu.astype(np.int64) * k + lab
    cnt = np.bincount(key, minlength=(cu.max() + 1) * k).reshape(-1, k)
    top = cnt[heavy].max(1) / cnt[heavy].sum(1)
    return top.mean()

print(f"{'k':>5s} {'top-cluster share':>18s} {'shuffled':>9s} {'ratio':>6s} {'1/k':>7s}")
for k in (5, 10, 20, 50, 100, 200):
    _, lab = kmeans2(V, k, seed=0, minit="++", iter=30)
    real = concentration(lab, k)
    shuf = np.mean([concentration(rng.permutation(lab), k) for _ in range(3)])
    print(f"{k:5d} {real:18.3f} {shuf:9.3f} {real / shuf:6.2f} {1 / k:7.3f}", flush=True)

# with the same clusters (k = 20): are a user's CANDIDATES drawn from the user's clusters?
k = 20
_, lab = kmeans2(V, k, seed=0, minit="++", iter=30)
item_lab = np.full(n_items, -1); item_lab[np.flatnonzero(ok)] = lab
ucnt = np.bincount(cu.astype(np.int64) * k + lab[idx[ci]], minlength=(cu.max() + 1) * k).reshape(-1, k)
fav = ucnt.argmax(1)                                     # each user's most-clicked cluster
ev_user = fit["user"]; cand = fit["item"].reshape(-1, 10)
evs = np.flatnonzero(np.isin(ev_user, heavy))
cl = item_lab[cand[evs]]; valid = cl >= 0
in_fav = (cl == fav[ev_user[evs]][:, None]) & valid
size_share = np.bincount(lab, minlength=k) / len(lab)
print(f"\nk={k}: candidates from the user's favourite cluster: {in_fav.sum() / valid.sum():.3f} "
      f"(cluster's share of items {np.mean(size_share[fav[heavy]]):.3f})")
lbl = fit["label"].reshape(-1, 10)[evs]
print(f"click rate on candidates in the favourite cluster {lbl[in_fav].mean():.4f} vs others "
      f"{lbl[valid & ~in_fav].mean():.4f}")
