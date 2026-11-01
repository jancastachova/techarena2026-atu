# Run from the repository root: .venv/bin/python experiments/lab/item_refit.py
# Needs data/cache/22_vectors_c7.npz (MF vectors of an MF trained on days < d, 1 seed).
#
# Rare candidates are the item-side weak spot (final LightGBM, half B: 4-10 clicks AUC 0.803, 31+
# 0.859; a rare clicked item ranks 2.06th of 10 on average vs 1.26th). The MF estimates all item
# vectors jointly with one global l2 and early stopping. Here each item's vector is RE-FITTED from its
# own impressions with the MF's vector as the prior mean: per item, a penalised logistic regression of
# "clicked" on the user's taste vector at the time (mean MF vector of their known clicks),
#     logit = c + b_i + hv . w_i,   penalty (lam / 2) |w_i - W_mf[i]|^2 + (lam_b / 2) b_i^2
# fitted jointly for all items (full-batch Adam on impressions before day 6). Day 6 is then scored
# with the refitted vectors; compared with the MF's own dot, overall and by candidate clicks, and in a
# blend with it.
import importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(ROOT / "data/cache/22_vectors_c7.npz")
W = z["W_6"].astype(np.float32); evd = z["ev_6"]; hv6 = z["hv_6"].astype(np.float32)
auc = e13.roc_auc
torch.set_num_threads(8)
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
tr_ev = np.flatnonzero(fit["day"] < 6)
HV = np.zeros((len(tr_ev), W.shape[1]), np.float32)
for s in range(0, len(tr_ev), 20000):
    e = tr_ev[s:s + 20000]
    H = hist[e]; m = (np.arange(H.shape[1])[None] < hlen[e][:, None]).astype(np.float32)
    HV[s:s + 20000] = (W[H] * m[:, :, None]).sum(1) / np.maximum(hlen[e], 1)[:, None]
rows_it = fit["item"].reshape(-1, 10)[tr_ev].ravel().astype(np.int64)
rows_y = fit["label"].reshape(-1, 10)[tr_ev].ravel().astype(np.float32)
rows_ev = np.repeat(np.arange(len(tr_ev)), 10)
X = torch.from_numpy(HV); It = torch.from_numpy(rows_it); Yt = torch.from_numpy(rows_y); Ev = torch.from_numpy(rows_ev)
W0 = torch.from_numpy(W)

def refit(lam, lam_b=1.0, passes=60):
    w = torch.nn.Parameter(W0.clone()); b = torch.nn.Parameter(torch.zeros(n_items)); c = torch.nn.Parameter(torch.tensor(-2.7))
    opt = torch.optim.Adam([w, b, c], lr=0.02)
    for p in range(passes):
        opt.zero_grad()
        loss = 0.0
        for s in range(0, len(It), 1_000_000):
            sl = slice(s, s + 1_000_000)
            q = c + b[It[sl]] + (X[Ev[sl]] * w[It[sl]]).sum(1)
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(q, Yt[sl], reduction="sum")
        loss = loss + 0.5 * lam * ((w - W0) ** 2).sum() + 0.5 * lam_b * (b ** 2).sum()
        loss.backward(); opt.step()
    return w.detach().numpy(), b.detach().numpy()

items6 = fit["item"].reshape(-1, 10)[evd]; y6 = fit["label"].reshape(-1, 10)[evd].ravel()
k6 = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
nclk = np.bincount(fit["clk_item"][k6], minlength=n_items)[items6].ravel()
mf_dot = np.einsum("ed,ekd->ek", hv6, W[items6]).ravel()
r6 = (evd[:, None] * 10 + np.arange(10)).ravel()
mf30 = ((np.load(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz")["mf_fit"]
         + np.load(ROOT / "data/cache/15_mf_c7_46f30c940a6a.npz")["mf_fit"]) / 2)[r6, 0]
zs = lambda s: (s - s.mean()) / s.std()
def report(name, s):
    parts = [f"{lo}-{hi}: {auc(s[(nclk >= lo) & (nclk <= hi)], y6[(nclk >= lo) & (nclk <= hi)]):.4f}"
             for lo, hi in [(0, 10), (11, 30), (31, 10**9)]]
    print(f"  {name:34s} AUC {auc(s, y6):.5f}   by candidate clicks {', '.join(parts)}", flush=True)
print("day 6, user vectors of the MF trained on days 0-5:")
report("MF dot (prior, 1 seed)", mf_dot)
report("MF30 full logit (30 seeds)", mf30)
for lam in [3.0, 30.0, 300.0]:
    w, b = refit(lam)
    dot = np.einsum("ed,ekd->ek", hv6, w[items6]).ravel()
    s = dot + b[items6].ravel()
    report(f"refit lam {lam:g}: dot only", dot)
    report(f"refit lam {lam:g}: dot + bias", s)
    report(f"  MF30 logit + refit (z-sum)", zs(mf30) + zs(s))
    print(f"    mean |w - W_mf| / mean |W_mf|: {np.abs(w - W).mean() / np.abs(W).mean():.3f}", flush=True)
