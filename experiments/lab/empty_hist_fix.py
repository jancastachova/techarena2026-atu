# Run from the repository root: TEST_DAY=6 .venv/bin/python experiments/lab/empty_hist_fix.py data/cache/box/lab/lab_preds_aud40a_d6.npz
# EMPTY-HISTORY FIX (2026-11-02; lab/gap_map.py: events with no known clicks rank BELOW random inside the event on days 6 / 5 / 4,
# 0.453 / 0.464 / 0.481, ~1.3% of evaluation). Fixed rule, nothing fitted: inside each such event the candidates are re-ordered by
# log(1 + item clicks before the day), centred and scaled to the event's own logit std; the event's mean logit is kept.
# Reports eval-weighted global AUC before / after with a paired bootstrap over half-B events.
import importlib.util, os, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
T = int(os.environ.get("TEST_DAY", 6))
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2)
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG); _, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
z = np.load(sys.argv[1]); key = [k for k in z.files if k.endswith("|B")][0]
r = rows[z["rows_B"]]; p = z[key].astype(np.float64); o = np.argsort(r, kind="stable"); r, p = r[o], p[o]
Y = fit["label"][r].reshape(-1, 10).astype(float); ev = r[::10] // 10; I = fit["item"][r].reshape(-1, 10)
L = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))).reshape(-1, 10)
B = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
def grp(h):
    g = np.zeros(len(h), int)
    for k, (lo, hi) in enumerate(B): g[(h >= lo) & (h <= hi)] = k
    return g
d_ev = np.flatnonzero(fit["day"] == T)
ratio = (np.bincount(grp(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(grp(hlen[d_ev].astype(int)), minlength=5) / len(d_ev))
W = ratio[grp(hlen[ev].astype(int))]
prev = np.repeat(fit["day"], 10) < T
pop = np.log1p(np.bincount(fit["item"][prev], weights=fit["label"][prev], minlength=n_items + 1))[I]
L2 = L.copy(); e0 = hlen[ev] == 0
z_ = pop[e0] - pop[e0].mean(1, keepdims=True); z_ /= np.maximum(z_.std(1, keepdims=True), 1e-9)
L2[e0] = L[e0].mean(1, keepdims=True) + z_ * L[e0].std(1, keepdims=True)
def wauc(s, y_, w_):
    o_ = np.argsort(s, kind="mergesort"); s, y_, w_ = s[o_], y_[o_], w_[o_]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w_ * y_); wn = np.bincount(g, w_ * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
a, b = wauc(L.ravel(), Y.ravel(), np.repeat(W, 10)), wauc(L2.ravel(), Y.ravel(), np.repeat(W, 10))
rng = np.random.default_rng(0); d = []
for _ in range(200):
    s = rng.integers(0, len(L), len(L)); w = np.repeat(W[s], 10)
    d.append(wauc(L2[s].ravel(), Y[s].ravel(), w) - wauc(L[s].ravel(), Y[s].ravel(), w))
print(f"day {T}: {e0.sum()} empty-history events; eval-weighted {a:.5f} -> {b:.5f}  B-A {b - a:+.5f}  "
      f"95% [{np.quantile(d, .025):+.5f}, {np.quantile(d, .975):+.5f}]")
