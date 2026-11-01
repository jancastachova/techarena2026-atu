# Run from the repository root: .venv/bin/python experiments/lab/taste_set.py
# Needs data/cache/22_vectors_c7.npz. How is taste built from the clicked items? Day 6, MF vectors of
# days 0-5: user vector = mean over (a) every known click (multiset, as the MF does), (b) UNIQUE clicked
# items, (c) unique items weighted by sqrt(times clicked), (d) all known clicks without the 100 cap.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
z = np.load(ROOT / "data/cache/22_vectors_c7.npz")
W = z["W_6"].astype(np.float64); evd = z["ev_6"]
items = fit["item"].reshape(-1, 10)[evd]; y = fit["label"].reshape(-1, 10)[evd].ravel()
H100, n100 = e13.long_history(fit, fit, 100)
Hall, nall = e13.long_history(fit, fit, 400)
H100, n100, Hall, nall = H100[evd], n100[evd].astype(int), Hall[evd], nall[evd].astype(int)
h = np.repeat(n100, 10)
groups = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21-50", (h >= 21) & (h <= 50)), ("51+", h >= 51)]
def sc(hv):
    s = np.einsum("ed,ekd->ek", hv, W[items]).ravel()
    return f"all {auc(s, y):.5f}  " + "  ".join(f"{g} {auc(s[k], y[k]):.4f}" for g, k in groups)
hv_multi = np.zeros((len(evd), W.shape[1])); hv_uniq = hv_multi.copy(); hv_sqrt = hv_multi.copy(); hv_all = hv_multi.copy()
for i in range(len(evd)):
    if n100[i]:
        its = H100[i, :n100[i]]
        hv_multi[i] = W[its].mean(0)
        u_, c_ = np.unique(its, return_counts=True)
        hv_uniq[i] = W[u_].mean(0)
        wt = np.sqrt(c_); hv_sqrt[i] = (W[u_] * wt[:, None]).sum(0) / wt.sum()
    if nall[i]:
        hv_all[i] = W[Hall[i, :nall[i]]].mean(0)
print(f"events with a repeated item in the 100: {np.mean([len(set(H100[i, :n100[i]])) < n100[i] for i in range(len(evd))]):.3f}; "
      f"with more than 100 known clicks: {np.mean(nall > 100):.3f}")
print("multiset mean (current)      ", sc(hv_multi))
print("unique items                 ", sc(hv_uniq))
print("unique, weight sqrt(count)   ", sc(hv_sqrt))
print("all known clicks (no 100 cap)", sc(hv_all))
