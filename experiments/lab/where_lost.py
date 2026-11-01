# Run from the repository root: .venv/bin/python experiments/lab/where_lost.py <preds.npz> <label>
# WHERE DOES THE GLOBAL AUC GO? Half-B predictions of a lab run split as log p = log S (event level, S = sum of the 10
# predictions) + log(p / S) (within event). Global AUC of the model, of each part alone, and with either part replaced by
# the truth (event part -> whether the event has a click; within part -> 1 for the clicked row, epsilon for the others),
# overall and by known clicks. Shows how much a perfect event model vs a perfect within-event ranker would add.
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
z = np.load(sys.argv[1]); rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)[z["rows_B"]]; p = z[f"{sys.argv[2]}|B"].astype(np.float64)  # rows_B indexes the day>=2 rows
y = fit["label"][rows].astype(np.float64)
P, Y = p.reshape(-1, 10), y.reshape(-1, 10)
ev = rows.reshape(-1, 10)[:, 0] // 10
H, n = e13.long_history(fit, fit, 100); nk = n[ev].astype(int)
S = P.sum(1, keepdims=True); W = P / S
yev = Y.max(1)
def g(s, m=None):
    m = np.ones(len(nk), bool) if m is None else m
    return auc(s[m].ravel(), Y[m].ravel())
GR = [("all", None), ("known 0", nk == 0), ("1-5", (nk >= 1) & (nk <= 5)), ("6-20", (nk >= 6) & (nk <= 20)), ("21+", nk >= 21)]
print(f"events {len(ev)}, event click share {yev.mean():.3f}; event-level AUC of S: {auc(S.ravel(), yev):.5f}; "
      f"within-event AUC (clicked events): {np.mean([auc(W[k], Y[k]) for k in np.flatnonzero(yev)]):.4f}")
for name, m in GR:
    mm = np.ones(len(nk), bool) if m is None else m
    within_true = np.where(Y > 0, 1.0, 1e-3)
    ev_true = (yev + 1e-3)[:, None]
    print(f"{name:8s} n={mm.sum():6d} model {g(P, m):.5f} | event part only {g(np.repeat(S, 10, 1), m):.5f} "
          f"within part only {g(W, m):.5f} | perfect event part {g(ev_true * W, m):.5f}  perfect within part {g(S * within_true, m):.5f}"
          f" | event AUC {auc(S[mm].ravel(), yev[mm]):.4f}")
