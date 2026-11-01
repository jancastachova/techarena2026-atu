# Run from the repository root: .venv/bin/python experiments/lab/user_side.py
# USER-side noise in cosine space. A light user's taste is the normalised mean of a few item vectors. The audience
# machinery gives an independent estimate: u2 = mean over the user's known clicks j of C_j = A_j - 0.5 B_j (unit clicker
# tastes, lab/aud_weighting.py). Shrink: user = unit(hv) + c / (n + c) x unit(u2), n = known clicks; candidate = unit
# vector (or its unit audience C_i). The dot-product version (lab/light_rare_ideas.py) failed; norms were noise there.
# Day 6, 3 vector seeds of the MF of days 0-5.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
files = [ROOT / "data/cache/22_vectors_c7.npz"] + [ROOT / f"data/cache/26_vectors_c7_s{s}.npz" for s in (1, 2)]
DAY = 6
evd = np.flatnonzero(fit["day"] == DAY); items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
H, n = e13.long_history(fit, fit, 100); H, n = H[evd], n[evd].astype(int)
m = (np.arange(100)[None] < n[:, None]).astype(np.float64)
k = fit["clk_ts"] < e13.TRAIN_START + DAY * e13.DAY
cu, ci = fit["clk_user"][k].astype(np.int64), fit["clk_item"][k].astype(np.int64)
nu = int(max(cu.max(), fit["user"].max())) + 1
ev05 = np.flatnonzero(fit["day"] < DAY)
su = np.repeat(fit["user"][ev05].astype(np.int64), 10); si = fit["item"].reshape(-1, 10)[ev05].ravel()
unit = lambda X: X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)
acc = {}
for f in files:
    W = unit(np.load(f)[f"W_{DAY}"].astype(np.float64)); W[0] = 0
    S = np.zeros((nu, W.shape[1])); np.add.at(S, cu, W[ci]); nc = np.bincount(cu, minlength=nu).astype(float)
    oc = unit((S[cu] - W[ci]) / np.maximum(nc[cu] - 1, 1)[:, None]); ok = nc[cu] > 1
    A = np.zeros_like(W); np.add.at(A, ci[ok], oc[ok]); A /= np.maximum(np.bincount(ci[ok], minlength=len(W)), 1)[:, None]
    tu = unit(S / np.maximum(nc, 1)[:, None])
    Bv = np.zeros_like(W); np.add.at(Bv, si, tu[su]); Bv /= np.maximum(np.bincount(si, minlength=len(W)), 1)[:, None]
    C = A - 0.5 * Bv
    hv = unit((W[H] * m[:, :, None]).sum(1))
    u2 = unit((C[H] * m[:, :, None]).sum(1))
    for cand_name, V in (("item", W), ("audience", unit(C))):
        for c in (0, 2, 5, 10, 20, 1e9):
            g = (1.0 if c == 1e9 else c / (n + c))[..., None] if c != 1e9 else 1.0
            uu = unit(hv + g * u2)
            key = f"cand {cand_name:8s} c={'inf' if c == 1e9 else c}"
            acc[key] = acc.get(key, 0) + np.einsum("ed,ekd->ek", uu, V[items]).ravel() / len(files)
h = np.repeat(n, 10); G = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21+", h >= 21)]
for kk, s in acc.items():
    print(f"{kk:26s} AUC {auc(s, y):.5f}  " + "  ".join(f"{g_} {auc(s[mm], y[mm]):.4f}" for g_, mm in G), flush=True)
