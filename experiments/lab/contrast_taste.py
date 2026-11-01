# Run from the repository root: .venv/bin/python experiments/lab/contrast_taste.py
# CONTRASTIVE TASTE: every click also says which 9 shown items LOST. The taste has always been the mean of the clicked items'
# vectors only. Here, per user, from their events of days 0-5 (labels of past fit events = their clicks, known as of day 6):
#   taste = mean(unit z_clicked) [warm-up clicks included]  - beta  * mean over clicked events of mean(unit z_losers)
#                                                           - gamma * mean over no-click events of mean(unit z_shown)
# z = 8-dim projected MF vectors of days 0-5 (W_6, 3 seeds, scores averaged). Day-6 events with >= 1 known click, cosine
# with the candidate, AUC overall and by known clicks. beta < 0 would mean shown items are a weak POSITIVE signal (the
# candidate policy is slightly taste-targeted: cos(candidate, taste) ~0.04).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
C = ROOT / "data/cache"
VECS = [C / "22_vectors_c7.npz", C / "26_vectors_c7_s1.npz", C / "26_vectors_c7_s2.npz"]
def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
NU = int(max(fit["user"].max(), fit["clk_user"].max())) + 1
items = fit["item"].reshape(-1, 10).astype(np.int64); lab = fit["label"].reshape(-1, 10)
eu = fit["user"].astype(np.int64); day = fit["day"]
tr = np.flatnonzero(day < 6); d6 = np.flatnonzero(day == 6)
clk_ev = tr[lab[tr].max(1) > 0]; nck_ev = tr[lab[tr].max(1) == 0]
warm = fit["clk_ts"] < e13.TRAIN_START
wu, wi = fit["clk_user"][warm].astype(np.int64), fit["clk_item"][warm].astype(np.int64)
n_pos = np.bincount(eu[clk_ev], minlength=NU) + np.bincount(wu, minlength=NU)
n_neg = np.bincount(eu[clk_ev], minlength=NU); n_0 = np.bincount(eu[nck_ev], minlength=NU)
u6 = eu[d6]; it6 = items[d6]; y6 = lab[d6].ravel()
keep = n_pos[u6] > 0
u6, it6, y6 = u6[keep], it6[keep], lab[d6][keep].ravel()
kn = np.repeat(n_pos[u6], 10)
GR = [("1-5", (kn >= 1) & (kn <= 5)), ("6-20", (kn >= 6) & (kn <= 20)), ("21+", kn >= 21)]
parts = []
for v in VECS:
    W = np.load(v)["W_6"].astype(np.float64)
    _, U = np.linalg.eigh(W[1:].T @ W[1:]); Z = unit(W @ U[:, ::-1][:, :8]); Z[0] = 0
    P = np.zeros((NU, 8)); N = np.zeros((NU, 8)); N0 = np.zeros((NU, 8))
    ci = items[clk_ev][lab[clk_ev] > 0]                                  # one click per clicked event
    np.add.at(P, eu[clk_ev], Z[ci]); np.add.at(P, wu, Z[wi])
    los = (Z[items[clk_ev]] * (lab[clk_ev] == 0)[:, :, None]).sum(1) / 9
    np.add.at(N, eu[clk_ev], los)
    np.add.at(N0, eu[nck_ev], Z[items[nck_ev]].mean(1))
    P /= np.maximum(n_pos, 1)[:, None]; N /= np.maximum(n_neg, 1)[:, None]; N0 /= np.maximum(n_0, 1)[:, None]
    parts.append((P[u6], N[u6], N0[u6], unit(Z[it6])))
def score(beta, gamma):
    s = 0
    for P, N, N0, Zc in parts:
        s = s + np.einsum("ed,ekd->ek", unit(P - beta * N - gamma * N0), Zc).ravel() / len(parts)
    return s
print(f"{len(u6)} day-6 events; known clicks (warm-up + days 0-5) 1-5 / 6-20 / 21+: "
      + " / ".join(f"{m.sum() // 10}" for _, m in GR), flush=True)
for beta, gamma in [(0, 0), (-0.5, 0), (-0.2, 0), (0.2, 0), (0.5, 0), (1.0, 0), (0, 0.5), (0, -0.5), (0.5, 0.5)]:
    s = score(beta, gamma)
    print(f"beta {beta:+.1f} gamma {gamma:+.1f}: AUC {auc(s, y6):.5f} | " + "  ".join(f"{g} {auc(s[m], y6[m]):.4f}" for g, m in GR), flush=True)
