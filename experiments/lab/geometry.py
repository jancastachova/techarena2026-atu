# Run from the repository root: .venv/bin/python experiments/lab/geometry.py
# Normalising vector LENGTHS gave the cosine break (34). Is the geometry noisy in other ways? Learned embeddings
# usually share a mean direction and a few dominant (often popularity-like) principal components ("all-but-the-top",
# Mu & Viswanath 2018) -- or, the other way round, noise in the weak directions (the spectrum is flat for ~10
# components, then falls): keep only the top k components. Day 6, the 10 saved vector seeds (MF of days 0-5), cosine after: centring on the mean item
# vector (impression-weighted), removing the top 1 / 2 / 4 principal components, full whitening.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
files = [ROOT / "data/cache/22_vectors_c7.npz"] + [ROOT / f"data/cache/26_vectors_c7_s{s}.npz" for s in range(1, 10)]
evd = np.flatnonzero(fit["day"] == 6); items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
H, n = e13.long_history(fit, fit, 100); H, n = H[evd], n[evd].astype(int)
m = (np.arange(100)[None] < n[:, None]).astype(np.float32)
w_imp = np.bincount(fit["item"][np.repeat(fit["day"] < 6, 10)], minlength=n_items + 1).astype(np.float64)
def cos_score(W):
    Wn = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    hv = (W[H] * m[:, :, None]).sum(1) / np.maximum(n, 1)[:, None]
    hv /= np.maximum(np.linalg.norm(hv, axis=1, keepdims=True), 1e-9)
    return np.einsum("ed,ekd->ek", hv, Wn[items]).ravel()
acc = {}
for f in files:
    W = np.load(f)["W_6"].astype(np.float64)
    mu = (W * w_imp[:len(W), None]).sum(0) / w_imp.sum()
    Wc = W - mu; Wc[0] = 0
    C = (Wc * w_imp[:len(W), None]).T @ Wc / w_imp.sum()
    ev_, U = np.linalg.eigh(C); U = U[:, ::-1]; ev_ = ev_[::-1]
    var = {"plain": W, "centred": Wc}
    for k in (6, 8, 10, 12, 14, 16, 20, 24):
        P = (W @ U[:, :k]) @ U[:, :k].T; P[0] = 0; var[f"keep top {k} PC"] = P
    for k_, V in var.items():
        acc[k_] = acc.get(k_, 0) + cos_score(V) / len(files)
    if f == files[0]:
        print("eigenvalue shares: " + ", ".join(f"{x:.3f}" for x in (ev_ / ev_.sum())) +
              f"   |mean| / mean |v| {np.linalg.norm(mu) / np.linalg.norm(W[1:], axis=1).mean():.3f}", flush=True)
h = np.repeat(n, 10); G = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21+", h >= 21)]
for k_, s in acc.items():
    print(f"{k_:22s} AUC {auc(s, y):.5f}  " + "  ".join(f"{g} {auc(s[mm], y[mm]):.4f}" for g, mm in G), flush=True)
