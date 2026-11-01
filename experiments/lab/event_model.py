# Run from the repository root (same arguments as lab/extra_member.py, --base-extra only):
#   .venv/bin/python experiments/lab/event_model.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# lab/decompose.py: on half B the row model orders candidates inside an event well (0.830) but
# ranks EVENTS ("does this event get a click") only at 0.692, and a perfect event level would lift
# global AUC to 0.909. Here a separate event-level LightGBM predicts "the event gets a click" from
# the event's features: the event-constant features, and per score column the event max, mean
# and log-sum-exp. Final row score = P(event click) x the row model's within-event share, and
# blends of that with the row model. Fit days 2-5; day 6 half A early stopping, half B AUC.
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
from scipy.stats import rankdata
p = {**e13.LGB_PARAMS, "learning_rate": 0.02}
es = lambda: [lgb.early_stopping(150, verbose=False)]
m = lgb.train(p, lgb.Dataset(base_X[tr], y[tr], feature_name=base_names), 8000,
              valid_sets=[lgb.Dataset(base_X[A], y[A])], callbacks=es())
prow = m.predict(base_X, num_iteration=m.best_iteration)          # in-sample on days 2-5
print(f"row model half-B {auc(prow[B], y[B]):.5f}", flush=True)

# event features: columns constant within every event stay as they are; others -> max, mean, lse
Xe3 = base_X.reshape(-1, 10, base_X.shape[1])
const = np.all(Xe3 == Xe3[:, :1], axis=(0, 1))
vary = ~const
mx, mn = Xe3[:, :, vary].max(1), Xe3[:, :, vary].mean(1)
vn = [n_ for n_, v in zip(base_names, vary) if v]
XE = np.concatenate([Xe3[:, 0, const], mx, mn], 1).astype(np.float32)
en = [n_ for n_, c in zip(base_names, const) if c] + [f"max_{n_}" for n_ in vn] + [f"mean_{n_}" for n_ in vn]
en = [f"f{j}_{n_}" for j, n_ in enumerate(en)]                  # base names repeat
yE = y.reshape(-1, 10).max(1)
trE, AE, BE = tr[::10], A[::10], B[::10]
print(f"event features: {XE.shape[1]} ({const.sum()} event-constant)", flush=True)
pe = {**p, "min_data_in_leaf": 100}
me = lgb.train(pe, lgb.Dataset(XE[trE], yE[trE], feature_name=en), 8000,
               valid_sets=[lgb.Dataset(XE[AE], yE[AE])], callbacks=es())
pE = me.predict(XE, num_iteration=me.best_iteration)
P = prow.reshape(-1, 10)
print(f"event-level AUC half B: event model {auc(pE[BE], yE[BE]):.5f}, row model sum "
      f"{auc(P.sum(1)[BE], yE[BE]):.5f}", flush=True)
g = sorted(zip(me.feature_importance("gain"), en), reverse=True)[:12]
print("  event model top features: " + ", ".join(n_ for _, n_ in g))

share = P / P.sum(1, keepdims=True)
yb = y[B]
def show(label, s):
    print(f"  {label:44s} half-B {auc(s.ravel()[B], yb):.5f}", flush=True)
print("\ncombinations (row score -> half-B global AUC):")
show("row model", P)
show("P(event) x share", pE[:, None] * share)
# keep the row model's own event level partly: geometric blend of event levels
lev_row = P.sum(1)
for wgt in [0.25, 0.5, 0.75]:
    lev = np.exp(wgt * np.log(pE) + (1 - wgt) * np.log(np.clip(lev_row, 1e-6, None)))
    show(f"event level = row^{1-wgt:.2f} * model^{wgt:.2f}, x share", lev[:, None] * share)
