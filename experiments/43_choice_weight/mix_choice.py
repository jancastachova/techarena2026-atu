# Run from the repository root:
#   .venv/bin/python experiments/43_choice_weight/mix_choice.py ENSEMBLE.csv CHOICE_test.npy OUT.csv [--w 0.2]
# Experiment 43 (2026-11-01): the 42 ensemble with the CHOICE member at weight 0.40 instead of 0.25. The 42 file is the equal
# rank average of 4 members (lgb, xgb, choice, shallow), so 0.8 x it + 0.2 x rank(choice) gives lgb / xgb / shallow 0.20 each
# and choice 0.40 exactly. CHOICE_test.npy is that build's cached choice-member prediction (24_ensemble, data/cache/
# 24_choice_full_s<seed>_test.npy), in the order of the base cache's test rows. Lab (lab/choice_weight.py): choice 0.35-0.45
# beats 0.25 on days 6, 5 and 4; whole ensembles on the never-used day 3 (lab/ens_compose.py): B (42) -> C (this) plain /
# eval- / gap-weighted +0.00014 / +0.00013 / +0.00018, all intervals above 0.
import argparse, csv, importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ap = argparse.ArgumentParser(); ap.add_argument("ens"); ap.add_argument("choice"); ap.add_argument("out")
ap.add_argument("--w", type=float, default=0.2); a = ap.parse_args()
spec = importlib.util.spec_from_file_location("e13", "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
_, test, eval_order, _, _ = e13.load_base("data/cache/13_base_c7_9c47a117c762.npz")
ch = np.load(a.choice); assert len(ch) == len(test["rowid"])
rc = dict(zip(test["rowid"].tolist(), (rankdata(ch) / len(ch)).tolist()))
with open(a.ens, newline="") as f:
    r = csv.reader(f); assert next(r) == ["rowid", "predicted_probability"]; ens = {k: float(v) for k, v in r}
assert set(ens) == set(rc)
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rowid", "predicted_probability"])
    for rid in eval_order.tolist():
        w.writerow([rid, f"{(1 - a.w) * ens[rid] + a.w * rc[rid]:.10g}"])
print("wrote", a.out)
