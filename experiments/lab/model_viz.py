# Run from the repository root: .venv/bin/python experiments/lab/model_viz.py
# Needs data/cache/lab_members_bundle_day6.npz (lab/ensemble_v5.py --save-members: the day-6 half-B
# predictions of the 26_ensemble_bundle members, trained on fit days 2-5 and never on day 6).
# Writes docs/model/best_model.png: the best model (rank average of LightGBM, XGBoost and the MLP, as
# submitted) against held-out day-6 half B, plain and reweighted to evaluation.csv's mix of users.
import importlib.util, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
ROOT = Path.cwd()
OUT = ROOT / "docs" / "model"; OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
z = np.load(ROOT / "data/cache/lab_members_bundle_day6.npz")
rows = np.flatnonzero(np.repeat(fit["day"], 10) >= 2); r = rows[z["rows_B"]]
y = fit["label"][r].astype(float)
names = ["LGB full (lr .02)", "XGB full", "MLP full"]
score = sum(rankdata(z[f"{k}|B"]) for k in names) / (3 * len(r))          # the submitted rank average
prob = np.mean([z[f"{k}|B"] for k in names], 0)                          # mean probability (calibration)
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG); _, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
BK = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 100)]
bucket = lambda h: np.digitize(h, [1, 6, 21, 51])
h = hlen[r // 10].astype(int); b = bucket(h)
e6 = np.flatnonzero(fit["day"] == 6)
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(bucket(hlen[e6].astype(int)), minlength=5) / len(e6))
w = ratio[b]
auc = e13.roc_auc
def roc(s, y_, w_):
    o = np.argsort(-s); s, y_, w_ = s[o], y_[o], w_[o]
    tp = np.cumsum(w_ * y_); fp = np.cumsum(w_ * (1 - y_))
    return fp / fp[-1], tp / tp[-1]
def wauc(s, y_, w_):
    fpr, tpr = roc(s, y_, w_); return float(np.trapezoid(tpr, fpr))

fig, ax = plt.subplots(2, 4, figsize=(19, 8.5))
fig.suptitle("Best model (26_ensemble_bundle: LightGBM + XGBoost + MLP, rank average) on held-out day-6 half B "
             f"({len(r) // 10:,} events, {len(r):,} rows)", fontsize=13)
# 1 ROC
a = ax[0, 0]
for lab_, ww, c in [("all day-6 events", np.ones_like(w), "C0"), ("reweighted to evaluation.csv's users", w, "C1")]:
    fpr, tpr = roc(score, y, ww); a.plot(fpr, tpr, c, label=f"{lab_}: AUC {wauc(score, y, ww):.4f}")
a.plot([0, 1], [0, 1], "k:", lw=0.8); a.set_xlabel("false positive rate"); a.set_ylabel("true positive rate")
a.set_title("ROC curve (public leaderboard: 0.8228)"); a.legend(fontsize=8, loc="lower right")
# 2 calibration
a = ax[0, 1]
q = np.quantile(prob, np.linspace(0, 1, 21)); qi = np.clip(np.digitize(prob, q[1:-1]), 0, 19)
pm = np.array([prob[qi == k].mean() for k in range(20)]); am = np.array([y[qi == k].mean() for k in range(20)])
a.plot(pm * 100, am * 100, "o-"); mx = max(pm.max(), am.max()) * 100
a.plot([0, mx], [0, mx], "k:", lw=0.8); a.set_xlabel("predicted click probability %"); a.set_ylabel("actual click rate %")
a.set_title("Calibration (20 equal-size bins)")
# 3 score distributions
a = ax[0, 2]
a.hist(score[y == 0], 50, alpha=0.6, density=True, label="not clicked"); a.hist(score[y == 1], 50, alpha=0.6, density=True, label="clicked")
a.set_xlabel("model score (rank, 0-1)"); a.set_title("Scores: clicked vs not clicked"); a.legend(fontsize=8)
# 4 AUC by known clicks
a = ax[0, 3]
lab = [f"{lo}-{hi}" if lo != hi else f"{lo}" for lo, hi in BK]
share_ev = np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)
vals = [auc(score[b == k], y[b == k]) if (b == k).sum() and 0 < y[b == k].mean() < 1 else np.nan for k in range(5)]
bars = a.bar(lab, vals, color="C2"); a.set_ylim(0.5, 0.9)
for k, br in enumerate(bars):
    a.text(br.get_x() + br.get_width() / 2, 0.51, f"{share_ev[k]:.0%}\nof eval", ha="center", fontsize=7)
a.set_xlabel("user's known clicks"); a.set_ylabel("AUC"); a.set_title("AUC by user history (share of evaluation events)")
# 5 AUC by candidate popularity
a = ax[1, 0]
k6 = fit["clk_ts"] < e13.TRAIN_START + 6 * e13.DAY
nc = np.bincount(fit["clk_item"][k6], minlength=n_items)[fit["item"][r]]
cb = np.digitize(nc, [4, 11, 31]); cl = ["0-3", "4-10", "11-30", "31+"]
a.bar(cl, [auc(score[cb == k], y[cb == k]) for k in range(4)], color="C3"); a.set_ylim(0.5, 0.9)
a.set_xlabel("candidate's clicks before the day"); a.set_ylabel("AUC"); a.set_title("AUC by candidate popularity")
# 6 rank of the clicked item in its event
a = ax[1, 1]
S = score.reshape(-1, 10); Y = y.reshape(-1, 10); has = Y.max(1) > 0
rk = (S[has] > S[has, Y[has].argmax(1)][:, None]).sum(1)
a.bar(range(1, 11), np.bincount(rk, minlength=10) / has.sum() * 100)
a.axhline(10, color="k", ls=":", lw=0.8); a.set_xlabel("model's rank of the clicked candidate (1 = top)")
a.set_ylabel("% of events with a click"); a.set_title(f"Within the event: clicked item ranked 1st in {np.mean(rk == 0):.0%}")
# 7 event level
a = ax[1, 2]
P = prob.reshape(-1, 10).sum(1); E = Y.max(1)
qe = np.quantile(P, np.linspace(0, 1, 11)); qk = np.clip(np.digitize(P, qe[1:-1]), 0, 9)
a.plot([P[qk == k].mean() * 100 for k in range(10)], [E[qk == k].mean() * 100 for k in range(10)], "o-")
a.plot([30, 95], [30, 95], "k:", lw=0.8)
a.set_xlabel("predicted P(event gets a click) %"); a.set_ylabel("actual share with a click %")
a.set_title(f"Event level: AUC {auc(P, E):.3f} (ceiling from propensity alone ~0.67)")
# 8 leaderboard progress
a = ax[1, 3]
txt = (ROOT / "experiments/RESULTS.md").read_text()
log = re.findall(r"^\| (2026-10-\d\d)[^|]*\| ([^|]+?) \| (0\.\d+) \|$", txt, re.M)
log = [(d, f, float(s_)) for d, f, s_ in log if float(s_) > 0.79]
best = np.maximum.accumulate([s_ for _, _, s_ in log])
a.plot(range(len(log)), [s_ for _, _, s_ in log], "o", ms=4, alpha=0.5, label="upload")
a.step(range(len(log)), best, where="post", label="best so far")
a.axhline(0.8262, color="C3", ls="--", lw=1, label="15th place (Phase 2), 0.8262")
a.axhline(0.8327, color="k", ls=":", lw=1, label="1st place, 0.8327")
a.set_xticks(range(0, len(log), 3), [log[i][0][5:] for i in range(0, len(log), 3)], rotation=45, fontsize=7)
a.set_ylabel("public AUC"); a.set_title("Public leaderboard progress"); a.legend(fontsize=7, loc="lower right")
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(OUT / "best_model.png", dpi=105)
print("wrote", OUT / "best_model.png")
