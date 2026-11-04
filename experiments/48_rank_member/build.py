# Experiment 48 (2026-11-03): 47s6 with a LIGHTGBM LAMBDARANK member (random 500-event groups, 2200 fixed trees, SET45,
# model seed 0; built on the fourth box by tools/box2_rank.sh -> submissions/48_rank_s0.csv). Weights from the day-6
# combination check (lab/combine_check.py: binary .375 / lambdarank .125 / choice .5 was +0.00004 [+0.00001, +0.00006]
# eval-weighted vs binary + choice): the three trees 0.125 each (were 1/6), lambdarank 0.125, choice SP + D 0.25 each as 47.
# Per seed s = 0..5: trees = 4 x 45_aud40_s<s> - rank(45 choice) (the three trees' ranks, as swap_choice.py), then the
# per-seed scores are averaged. A gamble the review asked for: expected about +0.00004, i.e. it answers "does the ranking
# member help" rather than closing the gap.
#     .venv/bin/python experiments/48_rank_member/build.py
import csv, importlib.util
import numpy as np
from scipy.stats import rankdata
spec = importlib.util.spec_from_file_location("e13", "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
_, test, eval_order, _, _ = e13.load_base("data/cache/13_base_c7_9c47a117c762.npz")
rid = test["rowid"].tolist(); n = len(rid); B = "data/cache/box"
def read(p):
    with open(p, newline="") as f:
        r = csv.reader(f); assert next(r) == ["rowid", "predicted_probability"]; d = {k: float(v) for k, v in r}
    return np.array([d[k] for k in rid])
rk = lambda v: rankdata(v) / n
r_rank = rk(read("submissions/48_rank_s0.csv"))
acc = np.zeros(n)
for s in range(6):
    trees = 4 * read(f"submissions/45_aud40_s{s}.csv") - rk(np.load(f"{B}/24_choice_full_45_s{s}_test.npy"))
    acc += 0.125 * trees + 0.25 * rk(np.load(f"{B}/24_choice_full_sp_s{s}_test.npy")) \
        + 0.25 * rk(np.load(f"{B}/24_choice_full_d45_s{s}_test.npy")) + 0.125 * r_rank
acc /= 6
out = dict(zip(rid, acc.tolist()))
with open("submissions/48_rank_member.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rowid", "predicted_probability"])
    for k in eval_order.tolist():
        w.writerow([k, f"{out[k]:.10g}"])
print("wrote submissions/48_rank_member.csv")
