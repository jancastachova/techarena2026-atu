# Run from the repository root: .venv/bin/python experiments/lab/aud_combo.py
# Do the noise fixes STACK? lab/aud_weighting.py (unit clicker tastes, +0.001) combined with lab/geometry.py's projection
# onto the top-k principal directions (k = 8 best for the plain cosine, +0.001), applied to the item vectors before
# everything else. Same day-6 setting.
# How should an item's AUDIENCE average its clickers? 26/34 take each clicker's mean taste (mean of the unit vectors
# of their OTHER clicks) with equal weight, so light clickers -- long, noisy taste vectors -- dominate the cosine.
# Day 6, cosine of the user (unit-vector mean of known clicks) with (A_i - B_i) for: clicker mean (current), click-
# weighted (every other click counts once), unit clicker tastes, sqrt(n)-weighted clicker means. Exposure B the same
# way over impressions. 3 vector seeds of the MF of days 0-5, scores averaged.
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
for PK in (0, 8, 12):
  for f in files:
      W0 = np.load(f)[f"W_{DAY}"].astype(np.float64)
      _, U = np.linalg.eigh(W0[1:].T @ W0[1:]); U = U[:, ::-1]
      W = unit((W0 @ U[:, :PK]) @ U[:, :PK].T) if PK else unit(W0); W[0] = 0
      S = np.zeros((nu, W.shape[1])); np.add.at(S, cu, W[ci]); nc = np.bincount(cu, minlength=nu).astype(float)
      other_sum = S[cu] - W[ci]; other_n = nc[cu] - 1; ok = other_n > 0
      taste = S / np.maximum(nc, 1)[:, None]
      hv = unit((W[H] * m[:, :, None]).sum(1))
      def agg(rows_i, vecs, wts):
          A = np.zeros_like(W); np.add.at(A, rows_i, vecs * wts[:, None])
          return A / np.maximum(np.bincount(rows_i, wts, minlength=len(W)), 1e-12)[:, None]
      oc = other_sum[ok] / other_n[ok][:, None]
      variants = {
          f"k={PK or 32} clicker mean": (agg(ci[ok], oc, np.ones(ok.sum())), agg(si, taste[su], np.ones(len(su)))),
          f"k={PK or 32} unit clickers": (agg(ci[ok], unit(oc), np.ones(ok.sum())), agg(si, unit(taste[su]), np.ones(len(su)))),
      }
      acc[(f"k={PK or 32} plain cosine", "W")] = acc.get((f"k={PK or 32} plain cosine", "W"), 0) + \
          np.einsum("ed,ekd->ek", hv, W[items]).ravel() / len(files)
      for name, (A, B) in variants.items():
          for lab, V in (("A", A), ("A-B", A - B), ("A-0.5B", A - 0.5 * B)):
              acc[(name, lab)] = acc.get((name, lab), 0) + np.einsum("ed,ekd->ek", hv, unit(V)[items]).ravel() / len(files)
h = np.repeat(n, 10); G = [("1-5", (h >= 1) & (h <= 5)), ("6-20", (h >= 6) & (h <= 20)), ("21+", h >= 21)]
for (name, lab), s in acc.items():
    print(f"{name:24s} {lab:7s} AUC {auc(s, y):.5f}  " + "  ".join(f"{g} {auc(s[mm], y[mm]):.4f}" for g, mm in G), flush=True)
