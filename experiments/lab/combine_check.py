# Run from the repository root: .venv/bin/python experiments/lab/combine_check.py
# 2026-11-03 (review: "is the final scoring function calibrated across events? does lambdarank add to the choice members?").
# Day-6 half-B predictions already on disk, all on the same rows: binary LightGBM, lambdarank G500 / G1000 (SET36,
# lab/global_rank2.py) and the choice model (full set, 3 seeds; lab/choice_model.py). Scores every fixed combination and three
# cross-event calibration schemes, evaluation-weighted (as lab/paired_ci.py), with a paired event bootstrap vs the current
# recipe (rank average of binary trees and choice).
import importlib.util
from pathlib import Path
import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
y_all = fit["label"][rows]
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
_, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
hl_row = hlen[rows // 10].astype(int)
BUCKETS = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def bucket(h):
    b = np.zeros(len(h), int)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(h >= lo) & (h <= hi)] = k
    return b
d6 = np.repeat(fit["day"], 10)[rows] == 6
b6 = bucket(hl_row[d6][::10])
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(b6, minlength=5) / len(b6))

C = ROOT / "data/cache"
G = np.load(C / "lab_preds_global_rank_set36.npz"); CH = np.load(C / "lab_preds_choice_full_s3_d6.npz")
rB = G["rows_B"]; assert np.array_equal(rB, CH["rows_B"])
y = y_all[rB].astype(float); w = ratio[bucket(hl_row[rB])]
M = {"bin": G["binary|B"], "r500": G["lambdarank G500|B"], "r1000": G["lambdarank G1000|B"],
     "ch": CH["base (= the best member)|B"]}
ev = np.unique(rB // 10, return_inverse=True)[1]
n_ev = ev.max() + 1
assert np.all(np.bincount(ev) == 10)


def wauc(s, y_, w_):
    o = np.argsort(s, kind="mergesort"); s, y_, w_ = s[o], y_[o], w_[o]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w_ * y_); wn = np.bincount(g, w_ * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


R = {k: rankdata(v) / len(v) for k, v in M.items()}
def ravg(**kw):
    return sum(wt * R[k] for k, wt in kw.items()) / sum(kw.values())
def lgt(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))
ev_sum = lambda p: np.bincount(ev, p)[ev]
share = lambda p: p / np.maximum(ev_sum(p), 1e-12)
pt = 1 / (1 + np.exp(-lgt(M["bin"])))                      # binary LightGBM probability
pc = M["ch"]                                               # choice probability (softmax incl. no click)

S = {
    "bin alone": M["bin"], "r500 alone": M["r500"], "ch alone": M["ch"],
    "BASE: rank avg bin + ch": ravg(bin=1, ch=1),
    "rank avg bin + r500": ravg(bin=1, r500=1),
    "rank avg r500 + ch": ravg(r500=1, ch=1),
    "rank avg bin + r500 + ch (1/3 each)": ravg(bin=1, r500=1, ch=1),
    "rank avg bin .25 r500 .25 ch .5": ravg(bin=1, r500=1, ch=2),
    "rank avg bin .375 r500 .125 ch .5": ravg(bin=3, r500=1, ch=4),
    "rank avg bin + r1000 + ch": ravg(bin=1, r1000=1, ch=1),
    "logit avg bin + ch": lgt(pt) + lgt(pc),
    "prob avg bin + ch": pt + pc,
    "event P(click) from ch x within share from bin": ev_sum(pc) * share(pt),
    "event P(click) from bin x within share from ch": ev_sum(pt) * share(pc),
    "geo: ch event x (bin share * ch share)^.5": ev_sum(pc) * np.sqrt(share(pt) * share(pc)),
}
base = S["BASE: rank avg bin + ch"]
rng = np.random.default_rng(0)
boots = [rng.integers(0, n_ev, n_ev) for _ in range(150)]
cnt = np.bincount  # event multiplicities per resample -> row weights
mult = [np.bincount(b, minlength=n_ev)[ev] for b in boots]
print(f"day 6 half B: {n_ev:,} events; eval-weighted AUC, paired bootstrap (150 event resamples) vs BASE")
b0 = [wauc(base, y, w * m) for m in mult]
for name, s in S.items():
    a = wauc(s, y, w)
    if name.startswith("BASE"):
        print(f"  {name:48s} {a:.5f}"); continue
    d = np.array([wauc(s, y, w * m) for m in mult]) - b0
    print(f"  {name:48s} {a:.5f}  vs BASE {a - wauc(base, y, w):+.5f} [{np.percentile(d, 2.5):+.5f}, {np.percentile(d, 97.5):+.5f}]",
          flush=True)
