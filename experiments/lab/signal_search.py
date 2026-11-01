# Run from the repository root: .venv/bin/python experiments/lab/signal_search.py
# Needs data/cache/lab_preds_slowlr.npz (the bundle-set LightGBM's day-6 half-B probabilities,
# lab/extra_member.py --save-preds) and the base cache.
#
# Do raw quantities the model does not see directly explain where it is wrong? ~15 per-row signals
# (all as of the event: earlier events, the click log before it, or statistics before day 6) are
# offered to a LightGBM that starts from the best model's own logit (init_score), fitted on half of
# half B's events and scored on the other half. Any AUC gain on the held-out events is information
# the best model lacks. Signals:
#   user: earlier events, distinct clicked items, repeat-click share, warm-up share of known clicks,
#         hours since the first known click, hours since the previous event, previous event clicked
#   candidate x user: times shown to this user before, shown in the user's previous event
#   item: clicks in the 24 h before day 6, days since its first click, share of its clicks that are
#         repeats, its click rate among repeat exposures, share of its clicks by light users
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path.cwd()
CACHE = ROOT / "data" / "cache"
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, n_users, n_items = e13.load_base(str(CACHE / "13_base_c7_9c47a117c762.npz"))
auc = e13.roc_auc
z = np.load(CACHE / "lab_preds_slowlr.npz")
rows_all = np.flatnonzero(np.repeat(fit["day"], 10) >= 2); rB = rows_all[z["rows_B"]]
p = np.clip(z["base (= the best member)|B"], 1e-6, 1 - 1e-6); y = fit["label"][rB].astype(float)
evB = rB // 10; items = fit["item"][rB].astype(np.int64)
T6 = e13.TRAIN_START + 6 * e13.DAY
hist, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
# --- user signals
u_ev = fit["user"][evB].astype(np.int64); ts_ev = fit["ts"][evB].astype(np.int64)
o = np.lexsort((fit["ts"], fit["user"])); us_o = fit["user"][o].astype(np.int64)
pos_in_user = np.empty(len(o), int); prev_ev = np.full(len(o), -1)
same = np.r_[False, us_o[1:] == us_o[:-1]]
c = 0
for i in range(len(o)):
    c = c + 1 if same[i] else 0
    pos_in_user[o[i]] = c
    if same[i]:
        prev_ev[o[i]] = o[i - 1]
n_prev_events = pos_in_user[evB]
pe = prev_ev[evB]
h_prev_event = np.where(pe >= 0, (ts_ev - fit["ts"][np.maximum(pe, 0)]) / 3600, np.nan)
prev_clicked = np.where(pe >= 0, fit["label"].reshape(-1, 10)[np.maximum(pe, 0)].max(1), np.nan)
H = hist[evB]; n = hlen[evB].astype(int); m = np.arange(H.shape[1])[None] < n[:, None]
n_distinct = np.array([len(set(H[i, :n[i]])) for i in range(len(H))])
rep_share = np.where(n > 0, 1 - n_distinct / np.maximum(n, 1), np.nan)
cu, cts = fit["clk_user"].astype(np.int64), fit["clk_ts"].astype(np.int64)
first_click = np.full(n_users + 1, np.iinfo(np.int64).max); np.minimum.at(first_click, cu, cts)
h_since_first = np.where(first_click[u_ev] < ts_ev, (ts_ev - first_click[u_ev]) / 3600, np.nan)
# AS OF THE EVENT: warm-up clicks / the user's clicks before the event (an earlier version divided by
# ALL the user's clicks, including later ones and the event's own click -- a label leak)
oc = np.lexsort((cts, cu)); key_c = cu[oc] * 10_000_000_000 + cts[oc]
known_before = np.searchsorted(key_c, u_ev * 10_000_000_000 + ts_ev) - np.searchsorted(key_c, u_ev * 10_000_000_000)
warm_before = np.searchsorted(key_c, u_ev * 10_000_000_000 + e13.TRAIN_START) - np.searchsorted(key_c, u_ev * 10_000_000_000)
warm_share = np.where(known_before > 0, warm_before / np.maximum(known_before, 1), np.nan)
# --- candidate x user: shown before to this user (earlier events, any day), shown in the previous event
shown_before = np.zeros(len(rB))
cnt = {}                                    # user -> item -> times shown so far (events in time order)
pos_rows = {int(r_): i for i, r_ in enumerate(rB)}
evB_set = set(evB.tolist())
for e in o:
    u = int(fit["user"][e]); its = fit["item"].reshape(-1, 10)[e].tolist(); d_u = cnt.setdefault(u, {})
    if e in evB_set:
        for k_, it in enumerate(its):
            shown_before[pos_rows[e * 10 + k_]] = d_u.get(it, 0)
    for it in its:
        d_u[it] = d_u.get(it, 0) + 1
prev_items = fit["item"].reshape(-1, 10)[np.where(pe >= 0, pe, 0)]          # (rows, 10)
in_prev = (items[:, None] == prev_items).any(1) & (pe >= 0)
# --- item signals (before day 6)
k6 = cts < T6
ci = fit["clk_item"].astype(np.int64)
clk24 = np.bincount(ci[k6 & (cts >= T6 - e13.DAY)], minlength=n_items)
first_item = np.full(n_items, np.iinfo(np.int64).max); np.minimum.at(first_item, ci[k6], cts[k6])
item_age = np.where(first_item < T6, (T6 - first_item) / e13.DAY, np.nan)
# repeat clicks of an item: a click on an item the same user clicked before
order_c = np.lexsort((cts, ci, cu)); key = cu[order_c] * n_items + ci[order_c]
is_rep = np.r_[False, key[1:] == key[:-1]]
rep_item = np.zeros(n_items); allc = np.zeros(n_items)
kk = k6[order_c]
np.add.at(rep_item, ci[order_c][kk], is_rep[kk]); np.add.at(allc, ci[order_c][kk], 1)
item_rep_share = np.where(allc > 0, rep_item / np.maximum(allc, 1), np.nan)
light_users = np.zeros(n_users + 1, bool)
ev_before = np.flatnonzero(fit["day"] < 6)
light_users[fit["user"][ev_before][hlen[ev_before] <= 20]] = True
lc = np.zeros(n_items); np.add.at(lc, ci[k6], light_users[cu[k6]])
item_light_share = np.where(allc > 0, lc / np.maximum(allc, 1), np.nan)
S = np.column_stack([n_prev_events, n_distinct, rep_share, warm_share, h_since_first, h_prev_event, prev_clicked,
                     shown_before, in_prev.astype(float), clk24[items], item_age[items], item_rep_share[items],
                     item_light_share[items]]).astype(np.float32)
names = ["user_prev_events", "user_distinct_clicked", "user_repeat_share", "user_warm_share", "user_h_since_first_click",
         "user_h_since_prev_event", "user_prev_event_clicked", "cand_times_shown_before", "cand_in_prev_event",
         "item_clicks_24h", "item_age_days", "item_repeat_share", "item_light_user_share"]
# residual model: start from the best model's logit, fit on half of half B's events, score the other half
rng = np.random.default_rng(3)
uev = np.unique(evB); side = np.zeros(evB.max() + 1, bool); side[rng.choice(uev, len(uev) // 2, replace=False)] = True
f1, f2 = side[evB], ~side[evB]
lg = np.log(p / (1 - p))
_, hl_eval = e13.long_history(test, fit, e13.HIST_LONG)
bucket = lambda h_: np.digitize(h_, [1, 6, 21, 51]); e6 = np.flatnonzero(fit["day"] == 6)
ratio = (np.bincount(bucket(hl_eval.astype(int)), minlength=5) / len(hl_eval)) / (np.bincount(bucket(hlen[e6].astype(int)), minlength=5) / len(e6))
w = ratio[bucket(n)]
def wauc(s, y_, w_):
    oo = np.argsort(s, kind="mergesort"); s, y_, w_ = s[oo], y_[oo], w_[oo]
    new = np.r_[True, s[1:] != s[:-1]]; g = np.cumsum(new) - 1
    wp = np.bincount(g, w_ * y_); wn = np.bincount(g, w_ * (1 - y_)); below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))
import lightgbm as lgb
P = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=15, min_data_in_leaf=500,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)
inner = np.zeros(len(rB), bool); inner[f1] = rng.random(f1.sum()) < 0.8
fa, fv = f1 & inner, f1 & ~inner
mdl = lgb.train(P, lgb.Dataset(S[fa], y[fa], init_score=lg[fa], feature_name=names), 3000,
                valid_sets=[lgb.Dataset(S[fv], y[fv], init_score=lg[fv])], callbacks=[lgb.early_stopping(100, verbose=False)])
adj = lg[f2] + mdl.predict(S[f2], num_iteration=mdl.best_iteration, raw_score=True)
print(f"held-out half of half B: best model {auc(lg[f2], y[f2]):.5f} / eval-weighted {wauc(lg[f2], y[f2], w[f2]):.5f}; "
      f"+ residual model on raw signals {auc(adj, y[f2]):.5f} / {wauc(adj, y[f2], w[f2]):.5f}  ({mdl.best_iteration} trees)")
gain = sorted(zip(mdl.feature_importance("gain"), names), reverse=True)
print("signals by gain: " + ", ".join(f"{n_} {g_:.0f}" for g_, n_ in gain))
r_ = y - p
for n_, col in zip(names, S.T):
    ok = ~np.isnan(col)
    print(f"  {n_:26s} corr with residual {np.corrcoef(rankdata(col[ok]), r_[ok])[0, 1]:+.4f}")
