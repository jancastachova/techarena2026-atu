# Run from the repository root: .venv/bin/python experiments/lab/replica_bench.py
# Bench for item-vector estimators on the calibrated replica (lab/replica_sim.py --scale 1.42 --repeat 0;
# our MF scores 0.8178 there vs 0.8175 on the real data). Item-side estimation costs ~0.041 AUC there
# (true items + pooled-click users 0.859 vs MF 0.818). Here, day 6, MF vectors trained on the replica's
# days 0-5 (vectors_pool --stage mf --save-vectors, seeds 0-2):
#   1. MF AUC and how well its scores track the TRUE taste logit, by candidate popularity
#   2. audience correction W' = W + gamma (A - beta B): A = mean taste of an item's clickers, B = mean
#      taste of the users it was shown to (tastes = pooled W over each user's clicks before day 6)
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/replica_s1.42_r0.npz"))
T = np.load(ROOT / "data/cache/replica_s1.42_r0_truth.npz")
auc = e13.roc_auc
seeds = [np.load(ROOT / f"data/cache/replica_vec_s{v}.npz") for v in range(3) if (ROOT / f"data/cache/replica_vec_s{v}.npz").exists()]
evd = seeds[0]["ev_6"]
items = fit["item"].reshape(-1, 10)[evd].astype(np.int64); y = fit["label"].reshape(-1, 10)[evd].ravel()
eu = fit["user"][evd].astype(np.int64)
true_logit = (np.einsum("ed,ekd->ek", T["theta"][eu].astype(np.float64), T["phi"][items].astype(np.float64))
              + T["b_item"][items]).ravel()
CUT = e13.TRAIN_START + 6 * e13.DAY
k = fit["clk_ts"] < CUT
pop = np.bincount(fit["clk_item"][k], minlength=n_items + 1)
cpop = pop[items].ravel()
PB = [("0-3", cpop <= 3), ("4-10", (cpop >= 4) & (cpop <= 10)), ("11-30", (cpop >= 11) & (cpop <= 30)), ("31+", cpop >= 31)]
H, n = e13.long_history(fit, fit, 100); H, n = H[evd], n[evd].astype(int)
mask = np.arange(100)[None, :] < n[:, None]

def pooled(W):
    hv = (W[H] * mask[..., None]).sum(1) / np.maximum(n, 1)[:, None]
    return hv

def score(W, b):
    return (np.einsum("ed,ekd->ek", pooled(W), W[items]) + b[items]).ravel()

def report(name, s):
    rows = "  ".join(f"{g} {auc(s[m], y[m]):.4f}/{np.corrcoef(s[m], true_logit[m])[0, 1]:.3f}" for g, m in PB)
    print(f"{name:34s} AUC {auc(s, y):.5f}  corr w/ true logit {np.corrcoef(s, true_logit)[0, 1]:.4f}  | by cand pop AUC/corr: {rows}", flush=True)

print(f"{len(seeds)} MF seed(s); share of rows by candidate clicks: " + ", ".join(f"{g} {m.mean():.3f}" for g, m in PB))
report("true taste logit", true_logit)
mf = np.mean([score(z["W_6"].astype(np.float64), z["b_6"].astype(np.float64)) for z in seeds], 0)
report("MF (seed-averaged scores)", mf)

# audience correction, per seed, in that seed's own space
cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
ev05 = np.flatnonzero(fit["day"] < 6)
iu = np.repeat(fit["user"][ev05].astype(np.int64), 10); ii = fit["item"].reshape(-1, 10)[ev05].ravel().astype(np.int64)
nu = int(max(cu.max(), fit["user"].max())) + 1
def audience(W):
    U = np.zeros((nu, W.shape[1])); c = np.bincount(cu, minlength=nu)
    np.add.at(U, cu, W[ci]); U /= np.maximum(c, 1)[:, None]
    A = np.zeros_like(W); np.add.at(A, ci, U[cu]); A /= np.maximum(np.bincount(ci, minlength=len(W)), 1)[:, None]
    Bv = np.zeros_like(W); np.add.at(Bv, ii, U[iu]); Bv /= np.maximum(np.bincount(ii, minlength=len(W)), 1)[:, None]
    return A, Bv
AB = [audience(z["W_6"].astype(np.float64)) for z in seeds]
for beta in (0.0, 0.5, 1.0):
    for gamma in (0.25, 0.5, 1.0, 2.0):
        s = np.mean([score(z["W_6"].astype(np.float64) + gamma * (A - beta * Bv), z["b_6"].astype(np.float64))
                     for z, (A, Bv) in zip(seeds, AB)], 0)
        report(f"W + {gamma} (A - {beta} B)", s)
