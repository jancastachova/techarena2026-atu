# Run from the repository root: TEST_DAY=6 NEW=data/cache/lab_preds_kern_all_d6.npz .venv/bin/python experiments/lab/kern_compare.py
# 2026-11-03: binary LightGBM on the lean features + an extra feature group (lab/lgb_choice.py --obj binary --extra ...) vs
# the same model without it (lab_preds_lgbc_binary_d{T}), alone and next to the neural choice member (box lab preds
# chC3x10, 10 seeds). Paired event bootstrap, evaluation- and gap-weighted as lab/paired_ci.py. (Copy of lgbc_compare.py.)
import importlib.util, os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
T = int(os.environ.get("TEST_DAY", 6))
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
cu, ct = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64); o_ = np.lexsort((ct, cu)); cu, ct = cu[o_], ct[o_]
def gap_bucket(src):                                       # as lab/paired_ci.py
    key = src["user"].astype(np.int64) * 10**10 + src["ts"].astype(np.int64)
    j = np.searchsorted(cu * 10**10 + ct, key, side="left") - 1; ok = (j >= 0) & (cu[np.maximum(j, 0)] == src["user"])
    g_ = np.full(len(key), np.inf); g_[ok] = (src["ts"][ok] - ct[j[ok]]) / 3600
    return np.searchsorted([0, 6, 12, 24, 48, 96, 192], g_, side="right") - 1
gb_fit, gb_ev = gap_bucket(fit), gap_bucket(test)


def wauc(s, y_, w):
    o = np.argsort(s, kind="mergesort"); s, y_, w = s[o], y_[o], w[o]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w * y_); wn = np.bincount(g, w * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


F = {"bin": f"data/cache/lab_preds_lgbc_binary_d{T}.npz", "gch": os.environ["NEW"],
     "nn": f"data/cache/box/lab/lab_preds_chC3x10_d{T}.npz"}
Z = {k: np.load(v) for k, v in F.items()}
rB = Z["bin"]["rows_B"]
assert all(np.array_equal(rB, z["rows_B"]) for z in Z.values())
R = {k: rankdata(z["base (= the best member)|B"]) / len(rB) for k, z in Z.items()}
y = y_all[rB].astype(float)
ev_ra = rows[rB][::10] // 10
d_ = np.repeat(fit["day"], 10)[rows] == T
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(bucket(hl_row[d_][::10]), minlength=5) / d_.sum() * 10)
gshare = np.bincount(gb_fit[ev_ra], minlength=7) / len(ev_ra)
W = {"eval": ratio[bucket(hl_row[rB])],
     "gap": np.repeat((np.bincount(gb_ev, minlength=7) / len(gb_ev) / np.maximum(gshare, 1e-12))[gb_fit[ev_ra]], 10)}
n_ev = len(rB) // 10
rng = np.random.default_rng(0)
cnt = [np.repeat(rng.multinomial(n_ev, np.full(n_ev, 1 / n_ev)), 10) for _ in range(150)]
S = {"gch alone": R["gch"], "bin + nn": .5 * R["bin"] + .5 * R["nn"], "gch + nn": .5 * R["gch"] + .5 * R["nn"]}
print(f"NEW = {os.environ['NEW']} (shown as gch)")
print(f"day {T} half B, {n_ev:,} events; eval- / gap-weighted AUC: bin alone {wauc(R['bin'], y, W['eval']):.5f} / "
      f"{wauc(R['bin'], y, W['gap']):.5f}, nn alone {wauc(R['nn'], y, W['eval']):.5f} / {wauc(R['nn'], y, W['gap']):.5f}")
for name, s in S.items():
    ref = R["bin"] if name == "gch alone" else S["bin + nn"]
    if s is ref:
        continue
    out = []
    for m, w in W.items():
        d = np.array([wauc(s, y, w * c) - wauc(ref, y, w * c) for c in cnt])
        out.append(f"{m} {wauc(s, y, w):.5f} ({wauc(s, y, w) - wauc(ref, y, w):+.5f} [{np.percentile(d, 2.5):+.5f}, {np.percentile(d, 97.5):+.5f}])")
    print(f"  {name:28s} vs {'bin' if name == 'gch alone' else 'bin + nn'}: " + "  ".join(out), flush=True)
