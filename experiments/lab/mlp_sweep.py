# Run from the repository root:
#   .venv/bin/python experiments/lab/mlp_sweep.py --seq <cache,...> --extra name=<cache> [...]
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and cached arrays in data/cache.
#
# The MLP is the best single member of the current ensemble (half B 0.83867 on the full feature
# set of 25_ensemble_impronly, vs LightGBM 0.83842) yet it was never tuned: 3 layers
# 256-128-64, dropout 0.1, lr 1e-3, 3 seeds, early stopping with patience 2. This sweeps width,
# depth, dropout, learning rate and the number of seeds. Training runs on the Apple GPU (mps)
# when available. Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B.
import argparse, importlib.util, time
from pathlib import Path
import numpy as np
from scipy.special import ndtri
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

def group(caches):
    S = np.mean([np.load(str(c))["mf_fit"] for c in caches.split(",")], 0)[rows]
    return np.concatenate([S, e13.relative_mf(S)], 1).astype(np.float32)

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
parts = [X30, POP, group(a.seq)] + [group(s_.split("=", 1)[1]) for s_ in a.extra]
XF = np.concatenate(parts, 1)

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

Z = np.zeros(XF.shape, np.float32); flags = []
for j in range(XF.shape[1]):
    col = XF[:, j].astype(np.float64); nan = np.isnan(col)
    ref = np.sort(col[tr & ~nan])
    q = (np.searchsorted(ref, col, side="left") + np.searchsorted(ref, col, side="right")) / 2
    Z[:, j] = np.where(nan, 0.0, ndtri(np.clip((q + 0.5) / (len(ref) + 1), 1e-4, 1 - 1e-4)))
    if nan.any():
        flags.append(nan.astype(np.float32))
if flags:
    Z = np.concatenate([Z] + [f[:, None] for f in flags], 1)
print(f"{len(y):,} rows, {Z.shape[1]} inputs", flush=True)

import torch
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_num_threads(4)
Zt = torch.from_numpy(Z).to(dev); yt = torch.from_numpy(y.astype(np.float32)).to(dev)
idx_tr, idx_A, idx_B = np.flatnonzero(tr), np.flatnonzero(A), np.flatnonzero(B)
tA, tB = torch.from_numpy(idx_A).to(dev), torch.from_numpy(idx_B).to(dev)

def predict(m, ti):
    m.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(m(Zt[ti[s:s + 65536]])).squeeze(1).cpu().numpy()
                               for s in range(0, len(ti), 65536)])

def run(widths, drop, lr, seed, batch=4096, patience=2, max_epochs=30):
    torch.manual_seed(seed); r = np.random.default_rng(seed)
    layers, w_in = [], Z.shape[1]
    for w in widths:
        layers += [torch.nn.Linear(w_in, w), torch.nn.ReLU(), torch.nn.Dropout(drop)]
        w_in = w
    m = torch.nn.Sequential(*layers, torch.nn.Linear(w_in, 1)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-6)
    best, state, bad = -1.0, None, 0
    for epoch in range(max_epochs):
        m.train()
        perm = torch.from_numpy(r.permutation(idx_tr)).to(dev)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Zt[b]).squeeze(1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        aa = auc(predict(m, tA), y[A])
        if aa > best:
            best, state, bad = aa, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return predict(m, tA), predict(m, tB), epoch + 1

configs = [
    ("current: 256-128-64, drop .1, lr 1e-3", (256, 128, 64), 0.1, 1e-3),
    ("wide: 512-256-128, drop .2", (512, 256, 128), 0.2, 1e-3),
    ("deep: 512-256-128-64, drop .1", (512, 256, 128, 64), 0.1, 1e-3),
    ("current, lr 3e-4", (256, 128, 64), 0.1, 3e-4),
    ("narrow: 128-64, drop .1", (128, 64), 0.1, 1e-3),
]
results = {}
for name, widths, drop, lr in configs:
    t0 = time.time(); pa, pb = [], []
    for seed in range(3):
        a_, b_, ep = run(widths, drop, lr, seed)
        pa.append(a_); pb.append(b_)
    results[name] = (pa, pb)
    print(f"  {name:40s} 3 seeds: half-A {auc(np.mean(pa, 0), y[A]):.5f}  half-B "
          f"{auc(np.mean(pb, 0), y[B]):.5f}  (single seeds half-B "
          + ", ".join(f"{auc(p, y[B]):.5f}" for p in pb) + f"; {time.time() - t0:.0f}s)", flush=True)

best_name = max(results, key=lambda k: auc(np.mean(results[k][0], 0), y[A]))   # chosen on half A
print(f"\nbest on half A: {best_name}; 10 seeds of it:", flush=True)
widths, drop, lr = [(w, dr, l) for n, w, dr, l in configs if n == best_name][0]
pa, pb = list(results[best_name][0]), list(results[best_name][1])
for seed in range(3, 10):
    a_, b_, _ = run(widths, drop, lr, seed)
    pa.append(a_); pb.append(b_)
    print(f"  {len(pb)} seeds: half-A {auc(np.mean(pa, 0), y[A]):.5f}  half-B {auc(np.mean(pb, 0), y[B]):.5f}",
          flush=True)
np.savez(CACHE / "lab_mlp_sweep_best.npz", A=np.mean(pa, 0), B=np.mean(pb, 0))
