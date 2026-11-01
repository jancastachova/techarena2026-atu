# Run from the repository root: .venv/bin/python experiments/lab/repeat_taste.py [replica base] [replica vectors s0,s1,s2]
# DOES TASTE DRIVE REPEAT CLICKS? The calibrated replica (lab/replica_fast.sh p12a15s125) matches the real data's cell AUCs
# (rms 0.0095) but not its repeat click rate: real 0.29 flat across user activity, replica 0.29 -> 0.45. Here, for repeat
# candidates on day 6 (candidate among the user's known clicks), click rate by projected-cosine quintile and by activity,
# real vs replica, plus the within-repeat AUC of the cosine. A flat real profile = repeats follow their own rule.
import importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
src = (ROOT / "experiments/lab/replica_cells.py").read_text()
g = {"__name__": "x"}
sys.argv = ["x", "data/cache/replica_p12a15s125.npz", "data/cache/replica_p12a15s125_vec_s0.npz"]
exec(src[:src.index("R = {name")], g)
C = ROOT / "data/cache"
worlds = [("real", C / "13_base_c7_9c47a117c762.npz", [C / "22_vectors_c7.npz", C / "26_vectors_c7_s1.npz", C / "26_vectors_c7_s2.npz"]),
          ("replica", C / "replica_p12a15s125.npz", [C / f"replica_p12a15s125_vec_s{s}.npz" for s in range(3)])]
for name, b, v in worlds:
    w = g["world"](b, v)
    rp = w["rep"]; s = w["cos"][rp]; y = w["y"][rp]; nn = np.repeat(w["nn"][:, None], 10, 1)[rp]
    q = np.quantile(s, [0.2, 0.4, 0.6, 0.8]); b_ = np.searchsorted(q, s)
    print(f"{name:8s} repeat rows {rp.sum()}  ctr {y.mean():.3f}  within-repeat AUC of cosine {e13.roc_auc(s, y):.4f}")
    print("   by cosine quintile: " + "  ".join(f"{y[b_ == k].mean():.3f}" for k in range(5)))
    for lo, hi in [(1, 5), (6, 20), (21, 50), (51, 10**9)]:
        m = (nn >= lo) & (nn <= hi)
        bb = b_[m]; yy = y[m]
        print(f"   known {lo}-{hi if hi < 10**9 else '+'}: ctr {yy.mean():.3f} | by quintile " + " ".join(f"{yy[bb == k].mean():.3f}" for k in range(5))
              + f" | AUC {e13.roc_auc(s[m], yy):.4f}", flush=True)
