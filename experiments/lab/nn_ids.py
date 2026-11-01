# Run from the repository root (same arguments as lab/extra_member.py, --base-extra = the full set):
#   .venv/bin/python experiments/lab/nn_ids.py --seq <cache,...> --base-extra name=<cache,...> [...]
# torch stage (this process) writes data/cache/lab_members_nnids_day6.npz; then
#   .venv/bin/python experiments/lab/nn_ids.py --blend    (no torch: blends it into the bundle members)
#
# Every member so far learns from precomputed scores only. This one also learns from raw IDs: the
# MLP member's inputs (Gaussian-rank features of the full set) plus a learned item embedding of
# the candidate (dim 16, plus an item bias), a SEPARATE embedding table averaged over the user's
# known clicks (up to 100), and their product, all in one network (256-128-64). It can learn
# item-specific and user-history-specific corrections that the scalar scores do not carry.
# Trained on fit days 2-5 rows only (time-forward like everything else), early stopping on half A of
# day 6; AUC on half B, plain and evaluation-weighted.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
OUT = CACHE / "lab_members_nnids_day6.npz"

if "--blend" in sys.argv:
    # blend into the 26_ensemble_bundle members (rank average), weight chosen on half A
    sys.argv = [sys.argv[0], str(CACHE / "lab_members_bundle_day6.npz"), str(CACHE / "lab_members_bundle_day6.npz")]
    src = (ROOT / "experiments/lab/paired_ci.py").read_text()
    ns = {"__name__": "x"}
    exec(src[:src.index("ones = np.ones")], ns)
    from scipy.stats import rankdata
    z, n = np.load(CACHE / "lab_members_bundle_day6.npz"), np.load(OUT)
    assert np.array_equal(z["rows_B"], n["rows_B"])
    y, w_eval, wauc = ns["y"], ns["w_eval"], ns["wauc"]
    rows = ns["rows"]; yA = ns["y_all"][z["rows_A"]]
    names = ["LGB full (lr .02)", "XGB full", "MLP full"]
    r = lambda p: rankdata(p) / len(p)
    baseA, baseB = sum(r(z[f"{k}|A"]) for k in names) / 3, sum(r(z[f"{k}|B"]) for k in names) / 3
    nnA, nnB = r(n["NNID|A"]), r(n["NNID|B"])
    wA = ns["ratio"][ns["bucket"](ns["hl_row"][z["rows_A"]])]
    print(f"NN with IDs alone: half-B {wauc(nnB, y, np.ones(len(y))):.5f}, eval-weighted {wauc(nnB, y, w_eval):.5f}; "
          f"bundle rank average {wauc(baseB, y, np.ones(len(y))):.5f} / {wauc(baseB, y, w_eval):.5f}; "
          f"rank corr {np.corrcoef(nnB, baseB)[0, 1]:.4f}")
    grid = np.round(np.arange(0, 0.61, 0.05), 2)
    wbest = grid[int(np.argmax([wauc((1 - w) * baseA + w * nnA, yA, wA) for w in grid]))]
    n_ev = len(y) // 10; rng = np.random.default_rng(0)
    counts = [np.repeat(rng.multinomial(n_ev, np.full(n_ev, 1 / n_ev)), 10) for _ in range(200)]
    for w in sorted({0.1, 0.2, 0.3, float(wbest)}):
        s = (1 - w) * baseB + w * nnB
        d = np.array([wauc(s, y, w_eval * c) - wauc(baseB, y, w_eval * c) for c in counts])
        lo, hi = np.percentile(d, [2.5, 97.5])
        tag = " (chosen on half A)" if w == wbest else ""
        print(f"  bundle x {1 - w:.2f} + NN-IDs x {w:.2f}{tag}: half-B {wauc(s, y, np.ones(len(y))):.5f}  eval-weighted "
              f"{wauc(s, y, w_eval):.5f}  {wauc(s, y, w_eval) - wauc(baseB, y, w_eval):+.5f} [{lo:+.5f}, {hi:+.5f}]")
    sys.exit(0)

src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
from scipy.special import ndtri
import torch
torch.set_num_threads(8)
dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
Z = np.zeros(base_X.shape, np.float32); flags = []
for j in range(base_X.shape[1]):
    col = base_X[:, j].astype(np.float64); nan = np.isnan(col)
    ref = np.sort(col[tr & ~nan])
    q = (np.searchsorted(ref, col, side="left") + np.searchsorted(ref, col, side="right")) / 2
    Z[:, j] = np.where(nan, 0.0, ndtri(np.clip((q + 0.5) / (len(ref) + 1), 1e-4, 1 - 1e-4)))
    if nan.any():
        flags.append(nan.astype(np.float32))
if flags:
    Z = np.concatenate([Z] + [f[:, None] for f in flags], 1)
item_row = fit["item"][rows].astype(np.int64)
Zt, yt = torch.from_numpy(Z), torch.from_numpy(y.astype(np.float32))
It, Ht = torch.from_numpy(item_row), torch.from_numpy(hist.astype(np.int64))
Evt = torch.from_numpy(ev.astype(np.int64))
EMB = 16

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ec = torch.nn.Embedding(n_items, EMB); self.eh = torch.nn.Embedding(n_items, EMB, padding_idx=0)
        self.bi = torch.nn.Embedding(n_items, 1)
        for e in (self.ec, self.eh):
            torch.nn.init.normal_(e.weight, std=0.01)
        torch.nn.init.zeros_(self.bi.weight)
        with torch.no_grad():
            self.eh.weight[0].zero_()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(Z.shape[1] + 3 * EMB, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    def forward(self, idx):
        idx_c = idx.cpu()
        z = Zt[idx_c].to(dev); it = It[idx_c].to(dev); h = Ht[Evt[idx_c]].to(dev)
        c = self.ec(it)
        n = (h > 0).sum(1, keepdim=True).clamp(min=1).float()
        u = self.eh(h).sum(1) / n
        self.reg = (c.pow(2).sum(1) + u.pow(2).sum(1)).mean()      # L2 on the embeddings of this batch
        return self.mlp(torch.cat([z, c, u, c * u], 1)).squeeze(1) + self.bi(it).squeeze(1)

idx_tr, idx_A, idx_B = np.flatnonzero(tr), np.flatnonzero(A), np.flatnonzero(B)
def predict(m, idx):
    m.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(m(torch.from_numpy(idx[s:s + 65536]))).cpu().numpy()
                               for s in range(0, len(idx), 65536)])
pa_all, pb_all = [], []
for seed in range(2):
    torch.manual_seed(seed); r = np.random.default_rng(seed)
    m = NN().to(dev)
    opt = torch.optim.Adam([{"params": m.mlp.parameters(), "weight_decay": 1e-6},
                            {"params": list(m.ec.parameters()) + list(m.eh.parameters()) + list(m.bi.parameters()),
                             "weight_decay": 0.0}], lr=1e-3)
    best, state, bad = -1.0, None, 0
    for epoch in range(20):
        m.train()
        perm = r.permutation(idx_tr)
        for s in range(0, len(perm), 4096):
            b = torch.from_numpy(perm[s:s + 4096])
            out = m(b)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yt[b].to(dev))
            loss = loss + 1e-4 * m.reg
            opt.zero_grad(); loss.backward(); opt.step()
        aa = auc(predict(m, idx_A), y[A])
        print(f"  seed {seed} epoch {epoch}: half-A {aa:.5f}", flush=True)
        if aa > best:
            best, state, bad = aa, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2:
                break
    m.load_state_dict(state)
    pa_all.append(predict(m, idx_A)); pb_all.append(predict(m, idx_B))
pa, pb = np.mean(pa_all, 0), np.mean(pb_all, 0)
print(f"NN with IDs: half-B {auc(pb, y[B]):.5f}  eval-weighted {wauc(pb, y[B], W_EVAL[B]):.5f}", flush=True)
np.savez(OUT, rows_A=np.flatnonzero(A), rows_B=np.flatnonzero(B), **{"NNID|A": pa, "NNID|B": pb})
