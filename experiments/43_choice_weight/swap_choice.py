# Run from the repository root:
#   .venv/bin/python experiments/43_choice_weight/swap_choice.py EQUAL4.csv OLD_CHOICE_test.npy NEW_CHOICE_test.npy OUT.csv
#       [--new-w 0.4] [--old-w 0.0]
# 2026-11-03: rebuild a 45-style file with another CHOICE member without retraining the trees. EQUAL4.csv is the build's
# equal rank average of lgb, xgb, choice, shallow (each rank / n), so 0.8 x EQUAL4 - 0.2 x rank(old) leaves lgb / xgb /
# shallow at 0.20 each; the choice weight 0.40 then goes to the new member (and optionally part of it to the old one).
import argparse, csv, importlib.util
import numpy as np
from scipy.stats import rankdata
ap = argparse.ArgumentParser(); ap.add_argument("ens"); ap.add_argument("old"); ap.add_argument("new"); ap.add_argument("out")
ap.add_argument("--new-w", type=float, default=0.4); ap.add_argument("--old-w", type=float, default=0.0)
ap.add_argument("--new2", help="a second new choice member (experiment 47: spectral + D45 choice, 0.2 each)")
ap.add_argument("--new2-w", type=float, default=0.0)
ap.add_argument("--tree-w", type=float, default=0.2, help="weight of each of the three tree members")
a = ap.parse_args()
assert abs(3 * a.tree_w + a.new_w + a.old_w + a.new2_w - 1) < 1e-9
spec = importlib.util.spec_from_file_location("e13", "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
_, test, eval_order, _, _ = e13.load_base("data/cache/13_base_c7_9c47a117c762.npz")
rid = test["rowid"].tolist()
r_old, r_new = (rankdata(np.load(p)) / len(rid) for p in (a.old, a.new))
with open(a.ens, newline="") as f:
    r = csv.reader(f); assert next(r) == ["rowid", "predicted_probability"]; ens = {k: float(v) for k, v in r}
e = np.array([ens[k] for k in rid])
s = a.tree_w * (4 * e - r_old) + a.old_w * r_old + a.new_w * r_new      # 4 e - r_old = the three trees' ranks
if a.new2:
    s = s + a.new2_w * rankdata(np.load(a.new2)) / len(rid)
out = dict(zip(rid, s.tolist()))
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rowid", "predicted_probability"])
    for k in eval_order.tolist():
        w.writerow([k, f"{out[k]:.10g}"])
print("wrote", a.out)
