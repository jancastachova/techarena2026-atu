# Run from the repository root: .venv/bin/python experiments/lab/gap_profile.py
# GAP PROFILE (2026-10-31): evaluation users are rarer visitors (median 42.6 h since their last click vs 24.6 h for lab day-6
# first events; eval events run to day 14). On day-6 half B of the 40 lab model: event click rate, AUC and the share of rows by
# hours-since-last-click bucket, for all day-6 events and for evaluation (no labels used from evaluation: only its gaps).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, _ = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
cu, ct = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64)
o = np.lexsort((ct, cu)); cu, ct = cu[o], ct[o]
def gap_h(user, ts):
    key = user.astype(np.int64) * 10**10 + ts.astype(np.int64)
    k2 = cu * 10**10 + ct
    j = np.searchsorted(k2, key, side="left") - 1                 # last click strictly before the event
    ok = (j >= 0) & (cu[np.maximum(j, 0)] == user)
    g = np.full(len(user), np.inf); g[ok] = (ts[ok] - ct[j[ok]]) / 3600; return g
g_fit = gap_h(fit["user"], fit["ts"]); g_ev = gap_h(test["user"], test["ts"])
BK = [0, 6, 12, 24, 48, 96, 192, np.inf]
bk = lambda g: np.minimum(np.searchsorted(BK, g, side="right") - 1, len(BK) - 2)
names = [f"{BK[i]}-{BK[i+1]}h" for i in range(len(BK) - 1)]; names[-1] = "192h+/none"
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
z = np.load("data/cache/lab_preds_40_d6.npz"); rB = z["rows_B"]; p = z["base (= the best member)|B"]
evB = rows[rB] // 10; y = fit["label"][rows[rB]]
bB = bk(g_fit[evB]); be = bk(g_ev)
d6 = np.flatnonzero(fit["day"] == 6); _, fi = np.unique(fit["user"][d6], return_index=True); bf = bk(g_fit[d6[fi]])
print(f"{'gap':12s} {'eval share':>10s} {'d6 share':>9s} {'d6 first':>9s} {'evt CTR':>8s} {'AUC in-bkt':>10s}")
for i, nm in enumerate(names):
    m = bB == i
    print(f"{nm:12s} {np.mean(be == i):10.4f} {np.mean(bB[::10] == i):9.4f} {np.mean(bf == i):9.4f} {y[m].mean() * 10:8.4f} {e13.roc_auc(p[m], y[m]) if m.any() and 0 < y[m].sum() < m.sum() else float('nan'):10.4f}")
from scipy.stats import rankdata
def wauc(s, y_, w):
    oo = np.argsort(s, kind="mergesort"); s, y_, w = s[oo], y_[oo], w[oo]
    new = np.r_[True, s[1:] != s[:-1]]; gg = np.cumsum(new) - 1
    wp = np.bincount(gg, w * y_); wn = np.bincount(gg, w * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
share_e = np.bincount(be, minlength=7) / len(be); share_b = np.bincount(bB, minlength=7) / len(bB)
w = (share_e / np.maximum(share_b, 1e-12))[bB]
print(f"half-B AUC plain {e13.roc_auc(p, y):.5f}  gap-weighted to eval {wauc(p, y, w):.5f}")

# DECOMPOSITION: within-event AUC (mean over clicked events) and AUC by gap inside known-clicks buckets, so a light-user effect
# is separated from a gap effect.
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG); hB = hlen[evB][::10]
P, Y, G = p.reshape(-1, 10), y.reshape(-1, 10), bB[::10]
clk = Y.sum(1) == 1
within = ((P[:, :, None] > P[:, None, :]) * Y[:, :, None] * (1 - Y[:, None, :])).sum((1, 2)) / 9
print("\nwithin-event AUC / event click rate / mean known clicks, by gap bucket")
for i, nm in enumerate(names):
    m = G == i
    print(f"  {nm:12s} within {within[m & clk].mean():.4f}  evt CTR {clk[m].mean():.4f}  known clicks {hB[m].mean():6.1f}")
print("\nrow AUC by gap bucket inside known-clicks groups (21-50, 51-100)")
for lo, hi in ((21, 50), (51, 100)):
    out = []
    for i, nm in enumerate(names):
        m = np.repeat((G == i) & (hB >= lo) & (hB <= hi), 10)
        if m.sum() > 2000: out.append(f"{nm} {e13.roc_auc(p[m], y[m]):.4f} (n={m.sum() // 10})")
    print(f"  {lo}-{hi}: " + " | ".join(out))
