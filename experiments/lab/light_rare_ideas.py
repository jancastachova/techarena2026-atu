# Run from the repository root: .venv/bin/python experiments/lab/light_rare_ideas.py
# Needs data/cache/22_vectors_c7.npz, data/cache/26ii_ii3_c7.npz, data/cache/26v5_hop2_c7.npz and the
# MF30 caches. Day-6 screen of ideas aimed at LIGHT USERS (few known clicks) and RARE CANDIDATES (few
# clicks), each alone and blended into the MF's own score, by known clicks and by candidate clicks:
#   L1 / R3  raw item-item co-clicks (26 item_item.py ii3, shrunk log-lift) -- full rank, item-specific
#   L2       history expansion scaled by sparsity: hv + lam / (n + k) * u2  (u2 = mean audience-minus-
#            exposure of the user's clicks, as 26's hop2), strongest for users with few clicks
#   L3       item click rate among LIGHT users only (<= 20 known clicks), from impressions before day 6
#   R1       rare items' vectors pulled toward their k-means cluster centre (k = 200, MF space), with
#            strength k_s / (k_s + clicks)
# Baseline = MF30's logit (30 seeds) for the blends; the 1-seed MF dot for the vector-based variants.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, n_users, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
z = np.load(ROOT / "data/cache/22_vectors_c7.npz")
W = z["W_6"].astype(np.float64); evd = z["ev_6"]; hv = z["hv_6"].astype(np.float64)
r6 = (evd[:, None] * 10 + np.arange(10)).ravel()
y = fit["label"][r6]; items = fit["item"].reshape(-1, 10)[evd]
mf30 = ((np.load(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz")["mf_fit"]
         + np.load(ROOT / "data/cache/15_mf_c7_46f30c940a6a.npz")["mf_fit"]) / 2)[r6, 0]
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
n = hlen[evd].astype(int); h = np.repeat(n, 10)
k6 = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
nclk_i = np.bincount(fit["clk_item"][k6], minlength=n_items).astype(float)
c = nclk_i[items].ravel()
zs = lambda s: (s - s.mean()) / s.std()
UG = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21+", h >= 21)]
CG = [("cand<=10", c <= 10), ("cand 11-30", (c > 10) & (c <= 30)), ("cand 31+", c > 30)]
def rep(name, s):
    parts = [f"{g} {auc(s[m], y[m]):.4f}" for g, m in UG + CG]
    print(f"  {name:40s} all {auc(s, y):.5f} | " + "  ".join(parts), flush=True)
def blend(name, s, ws=(0.1, 0.2, 0.4)):
    for w in ws:
        rep(f"MF30 + {w} x {name}", zs(mf30) + w * zs(s))
mf_dot = np.einsum("ed,ekd->ek", hv, W[items]).ravel()
print("day 6 baselines:")
rep("MF30 logit (30 seeds)", mf30)
rep("MF dot (1 seed)", mf_dot)

print("\nL1 / R3: raw item-item co-clicks (ii3)")
ii3 = np.load(ROOT / "data/cache/26ii_ii3_c7.npz")["mf_fit"][r6, 0]
rep("ii3 alone", ii3); blend("ii3", ii3)

print("\nL2: history expansion scaled by sparsity, hv + lam/(n+k) * u2 (1-seed vectors)")
# u2 = mean over the user's clicks of (audience - exposure) of each clicked item, as 26's hop2c input
src = (ROOT / "experiments/26_audience/audience.py").read_text()
ns_ = {"e13": e13, "np": np}
exec(src[src.index("def tastes"):src.index("def scores")], ns_)
A, na, Bx = ns_["audience"](fit, W, 6)
Cx = A - Bx
H = hist[evd]; msk = (np.arange(H.shape[1])[None] < n[:, None])[:, :, None]
u2 = (Cx[H] * msk).sum(1) / np.maximum(n, 1)[:, None]
for lam, k in [(1.0, 1), (3.0, 1), (3.0, 5), (10.0, 5)]:
    hv2 = hv + (lam / (n + k))[:, None] * u2
    rep(f"MF dot with expansion lam {lam:g} k {k}", np.einsum("ed,ekd->ek", hv2, W[items]).ravel())

print("\nL3: item click rate among light users (<= 20 known clicks) vs all users, impressions before day 6")
before = np.flatnonzero(fit["day"] < 6)
light_ev = before[hlen[before] <= 20]
def item_rate(evs):
    it = fit["item"].reshape(-1, 10)[evs].ravel(); yy = fit["label"].reshape(-1, 10)[evs].ravel()
    imp = np.bincount(it, minlength=n_items); clk = np.bincount(it, yy, minlength=n_items)
    p0 = clk.sum() / imp.sum()
    p = (clk + 20 * p0) / (imp + 20)                     # shrunk toward the overall rate
    return np.log(p / (1 - p))
rl, ra = item_rate(light_ev)[items].ravel(), item_rate(before)[items].ravel()
rep("item rate, all users", ra); rep("item rate, light users only", rl)
blend("light-user item rate", rl); blend("light minus overall item rate", rl - ra, ws=(0.1, 0.2))

print("\nR1: rare items' vectors shrunk toward their k-means cluster centre (k = 200)")
rng = np.random.default_rng(0)
cent = W[rng.choice(n_items, 200, replace=False)].copy()
for _ in range(15):
    lab = np.empty(n_items, int)
    for s in range(0, n_items, 10000):
        lab[s:s + 10000] = ((W[s:s + 10000, None] - cent[None]) ** 2).sum(2).argmin(1)
    for j in range(200):
        if (lab == j).any():
            cent[j] = W[lab == j].mean(0)
for ks in [2.0, 5.0, 20.0]:
    a_ = ks / (ks + nclk_i)
    Ws = (1 - a_)[:, None] * W + a_[:, None] * cent[lab]
    rep(f"MF dot, cluster-shrunk vectors k_s {ks:g}", np.einsum("ed,ekd->ek", hv, Ws[items]).ravel())
