# Run from the repository root: .venv/bin/python experiments/lab/segment_headroom.py
# Needs data/cache/lab_members_bundle_day6.npz (the 26_ensemble_bundle members' day-6 half-B predictions).
#
# How much global AUC could fixing one segment give AT MOST? For each segment, the best model's half-B
# scores are changed only inside that segment so that it is ranked PERFECTLY within its events, the
# rest untouched, and the evaluation-weighted AUC is recomputed:
#   user segments (events of users with 0 / 1-5 / 6-20 known clicks, new users without earlier train
#   events): in those events the clicked candidate gets the event's top score + a small margin, the
#   others keep theirs (perfect within-event order, unchanged event level)
#   candidate segment (candidates with <= 10 clicks): a rare candidate that was clicked gets its
#   event's top score + margin, a rare one that was not clicked gets its event's lowest score
# Upper bounds: no real fix ranks a segment perfectly. Also: a 30% closing of the gap to perfect.
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(ROOT / "data/cache/lab_members_bundle_day6.npz")
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2); r = rows[z["rows_B"]]
y = fit["label"][r].astype(float)
s0 = sum(rankdata(z[f"{k}|B"]) for k in ["LGB full (lr .02)", "XGB full", "MLP full"]) / (3 * len(r))
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG); _, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
bucket = lambda h: np.digitize(h, [1, 6, 21, 51])
e6 = np.flatnonzero(fit["day"] == 6)
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(bucket(hlen[e6].astype(int)), minlength=5) / len(e6))
h = hlen[r // 10].astype(int); w = ratio[bucket(h)]
def wauc(s, y_, w_):
    o = np.argsort(s, kind="mergesort"); s, y_, w_ = s[o], y_[o], w_[o]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w_ * y_); wn = np.bincount(g, w_ * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
base = wauc(s0, y, w)
S0, Y = s0.reshape(-1, 10), y.reshape(-1, 10)
ev = r.reshape(-1, 10)[:, 0] // 10; hev = hlen[ev]
first_ts = {}
for e in np.flatnonzero(fit["day"] < 6):
    first_ts.setdefault(int(fit["user"][e]), True)
newu = np.array([int(fit["user"][e]) not in first_ts for e in ev])
k6 = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
nc = np.bincount(fit["clk_item"][k6], minlength=n_items)[fit["item"][r]].reshape(-1, 10)
def user_fix(mask, frac=1.0):
    S = S0.copy(); top = S.max(1)
    idx = np.flatnonzero(mask & (Y.max(1) > 0)); j = Y[idx].argmax(1)
    S[idx, j] = S[idx, j] + frac * (top[idx] + 1e-4 - S[idx, j])
    return S.ravel()
def item_fix(frac=1.0):
    S = S0.copy(); top, bot = S.max(1, keepdims=True), S.min(1, keepdims=True)
    rare = nc <= 10
    up = rare & (Y > 0); dn = rare & (Y == 0)
    S = np.where(up, S + frac * (top + 1e-4 - S), S); S = np.where(dn, S - frac * (S - bot + 1e-4), S)
    return S.ravel()
print(f"best model, half B, evaluation-weighted AUC: {base:.5f}")
print(f"{'segment':34s} {'share of eval events':>21s}  {'perfect':>8s} {'gain':>8s}  {'30% of the way':>15s}")
share_ev = np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)
newu_eval = np.mean(~np.isin(test["user"], fit["user"]))
for name, mask, sh in [("users with 0 known clicks", hev == 0, share_ev[0]),
                       ("users with 1-5 known clicks", (hev >= 1) & (hev <= 5), share_ev[1]),
                       ("users with 6-20 known clicks", (hev >= 6) & (hev <= 20), share_ev[2]),
                       ("users with 21+ known clicks", hev >= 21, share_ev[3] + share_ev[4]),
                       ("new users (no earlier train events)", newu, newu_eval)]:
    p1, p3 = wauc(user_fix(mask), y, w), wauc(user_fix(mask, 0.3), y, w)
    print(f"{name:34s} {sh:>20.1%}  {p1:8.5f} {p1 - base:+8.5f}  {p3 - base:+15.5f}   ({mask.sum():,} half-B events)")
ce = np.bincount(np.minimum(np.bincount(fit['clk_item'][fit['clk_ts'] < e13.TRAIN_START + 7 * e13.DAY], minlength=n_items)[test['item']], 11), minlength=12)
p1, p3 = wauc(item_fix(), y, w), wauc(item_fix(0.3), y, w)
print(f"{'candidates with <= 10 clicks':34s} {ce[:11].sum() / ce.sum():>17.1%} rows  {p1:8.5f} {p1 - base:+8.5f}  {p3 - base:+15.5f}")
