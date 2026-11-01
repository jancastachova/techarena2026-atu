# Run from the repository root (same arguments as lab/extra_member.py, --base-extra = the full set):
#   .venv/bin/python experiments/lab/tf_stack.py --seq <cache,...> --base-extra name=<cache,...> [...]
#
# A time-forward STACK. Our earlier stacker (lab/output_blend.py) was fitted on day-6 half A only
# (~245k rows) over members trained on days 2-5, and was worse. Here level 1 gets honest out-of-time
# predictions for EVERY day 3-6: for each day d, a LightGBM on the full feature set (bundle set)
# trained on days 2..d-1 (trees by early stopping on a 10% event holdout of its own training days)
# predicts day d. Level 2 is a small LightGBM trained on days 3-5 with the
# level-1 prediction (logit), its rank and gap to the event's best, and key raw features (MF30 logit,
# audience / hop scores, known clicks, user click rate and events, item clicks, repeat flag), so it
# can learn when and for whom to trust level 1 -- from three days of rows instead of half a day.
# Day 6: half A early stopping of level 2, half B reported (plain, evaluation-weighted) against the
# level-1 model of day 6 alone (= the current best LightGBM member).
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
src = (ROOT / "experiments/lab/extra_member.py").read_text()
exec(src[:src.index("import lightgbm as lgb")])
import lightgbm as lgb
P = {**e13.LGB_PARAMS, "learning_rate": 0.03}
L1 = np.full(len(y), np.nan)
rng = np.random.default_rng(1)
for day in range(3, 7):
    trd = (d >= 2) & (d < day)
    ev_tr = np.unique(ev[trd]); hold = np.zeros(ev.max() + 1, bool)
    hold[rng.choice(ev_tr, len(ev_tr) // 10, replace=False)] = True
    fit_m, es_m = trd & ~hold[ev], trd & hold[ev]
    m = lgb.train(P, lgb.Dataset(base_X[fit_m], y[fit_m], feature_name=base_names), 6000,
                  valid_sets=[lgb.Dataset(base_X[es_m], y[es_m])], callbacks=[lgb.early_stopping(100, verbose=False)])
    k = d == day
    L1[k] = m.predict(base_X[k], num_iteration=m.best_iteration, raw_score=True)
    print(f"level 1 for day {day} (trained on days 2-{day - 1}, {m.best_iteration} trees): AUC {auc(L1[k], y[k]):.5f}", flush=True)
g = L1.reshape(-1, 10)
rel = np.stack([(g - g.max(1, keepdims=True)).ravel(), rankdata(-g, axis=1).ravel()], 1)
KEY = ["mf_logit", "audc_logit", "hop2c_logit", "hop3_logit", "seq_logit", "hist_len", "user_rate_logit",
       "user_log_events", "item_log_clicks", "rep_any"]
X2 = np.column_stack([L1, rel] + [base_X[:, base_names.index(n_)] for n_ in KEY]).astype(np.float32)
names2 = ["l1_logit", "l1_minus_max", "l1_rank"] + KEY
tr2 = (d >= 3) & (d <= 5)
P2 = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500,
          feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)
m2 = lgb.train(P2, lgb.Dataset(X2[tr2], y[tr2], feature_name=names2), 4000,
               valid_sets=[lgb.Dataset(X2[A], y[A])], callbacks=[lgb.early_stopping(150, verbose=False)])
p2 = m2.predict(X2[B], num_iteration=m2.best_iteration)
yb, wb = y[B], W_EVAL[B]
print(f"\nday 6 half B: level 1 alone half-B {auc(L1[B], yb):.5f}  eval-weighted {wauc(L1[B], yb, wb):.5f}")
print(f"              level 2 stack    half-B {auc(p2, yb):.5f}  eval-weighted {wauc(p2, yb, wb):.5f}  ({m2.best_iteration} trees)")
print("level 2 features by gain: " + ", ".join(n_ for _, n_ in sorted(zip(m2.feature_importance("gain"), names2), reverse=True)))
np.savez(str(CACHE / "lab_preds_tf_stack.npz"), rows_B=np.flatnonzero(B), **{"level1|B": L1[B], "stack|B": p2})
