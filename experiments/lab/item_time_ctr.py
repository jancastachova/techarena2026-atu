# Run from the repository root: .venv/bin/python experiments/lab/item_time_ctr.py
# Loads experiment 13 (experiments/13_retune_7days/retune.py) and its cached arrays in data/cache.
# Base features hold each item's overall click rate AND the event's 6-hour time bucket, but never
# their INTERACTION: with 50,000 items LightGBM cannot learn a per-item, per-bucket rate from
# splits. Here that aggregate is built from the cached arrays, as of each day (counts from days
# STRICTLY BEFORE the scored day, like the daily flush in the real pipeline), and added as
# features. Fit days 2-5; day 6 split by event: half A for early stopping, AUC on half B
# (baseline 0.83571).
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, _, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
mf = np.load(str(ROOT / "data/cache/13_mf_c7_bb1fcb0a9c89.npz"))["mf_fit"]
auc = e13.roc_auc
K = 50.0                                      # pseudo-count, shrinking toward the item's own rate
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

row_item = fit["item"]
row_day = np.repeat(fit["day"], 10)
row_lab = fit["label"].astype(np.float64)
bucket = ((fit["ts"] // 3600) % 24) // 6      # 6-hour UTC block, as in the data's time_bucket
row_bucket = np.repeat(bucket, 10)
print("rows per bucket:", np.bincount(row_bucket, minlength=4) / len(row_bucket))

rows = np.flatnonzero(row_day >= 2)
_, X, names = e13.feature_matrix(fit, mf, rows)
y, d, ev = fit["label"][rows], row_day[rows], rows // 10
NEW = np.zeros((len(rows), 3), np.float32)
pos = np.full(len(row_day), -1); pos[rows] = np.arange(len(rows))
key = row_item.astype(np.int64) * 4 + row_bucket
g_prior = e13.CTR_PRIOR
for day in range(2, 7):
    past = row_day < day                                      # only days strictly before
    impr_i = np.bincount(row_item[past], minlength=n_items).astype(np.float64)
    clk_i = np.bincount(row_item[past], weights=row_lab[past], minlength=n_items)
    ctr_i = (clk_i + K * g_prior) / (impr_i + K)
    impr_ib = np.bincount(key[past], minlength=n_items * 4).astype(np.float64)
    clk_ib = np.bincount(key[past], weights=row_lab[past], minlength=n_items * 4)
    ctr_ib = (clk_ib + K * np.repeat(ctr_i, 4)) / (impr_ib + K)   # shrunk to the item's own rate
    sel = np.flatnonzero(row_day == day)
    k, it = key[sel], row_item[sel]
    NEW[pos[sel], 0] = logit(ctr_ib[k])
    NEW[pos[sel], 1] = logit(ctr_ib[k]) - logit(ctr_i[it])
    NEW[pos[sel], 2] = np.log1p(impr_ib[k])

rng = np.random.default_rng(0)
ev6 = np.unique(ev[d == 6]); a_set = np.zeros(len(fit["day"]), bool)
a_set[rng.choice(ev6, len(ev6) // 2, replace=False)] = True
tr, A, B = d <= 5, (d == 6) & a_set[ev], (d == 6) & ~a_set[ev]

# does the interaction exist at all? click rate by bucket for the most-shown items, day 6 vs before
past = row_day < 6
impr_ib = np.bincount(key[past], minlength=n_items * 4).reshape(-1, 4)
clk_ib = np.bincount(key[past], weights=row_lab[past], minlength=n_items * 4).reshape(-1, 4)
busy = np.flatnonzero(impr_ib.min(1) >= 30)
r = (clk_ib[busy] + 1) / (impr_ib[busy] + 16)
spread = r.max(1) / r.min(1)
print(f"items with >=30 impressions in every bucket: {len(busy):,}; median max/min bucket rate "
      f"{np.median(spread):.2f}")
day6 = row_day == 6
i6 = np.bincount(key[day6], minlength=n_items * 4).reshape(-1, 4)
c6 = np.bincount(key[day6], weights=row_lab[day6], minlength=n_items * 4).reshape(-1, 4)
ok = np.flatnonzero((impr_ib.min(1) >= 30) & (i6.min(1) >= 10))
if len(ok) > 50:
    a = ((clk_ib[ok] + 1) / (impr_ib[ok] + 16)).ravel()
    b = ((c6[ok] + 1) / (i6[ok] + 16)).ravel()
    print(f"stability of item x bucket rates (before day 6 vs day 6, {len(ok)} items): "
          f"correlation {np.corrcoef(a, b)[0, 1]:.3f}")

import lightgbm as lgb
def run(label, Xm, nm):
    m = lgb.train(e13.LGB_PARAMS, lgb.Dataset(Xm[tr], y[tr], feature_name=nm), 3000,
                  valid_sets=[lgb.Dataset(Xm[A], y[A])], callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"  {label:34s} half-A {m.best_score['valid_0']['auc']:.5f}  half-B "
          f"{auc(m.predict(Xm[B], num_iteration=m.best_iteration), y[B]):.5f}  ({m.best_iteration} trees)",
          flush=True)
run("13 as is", X, names)
run("+ item x time-bucket click rate", np.concatenate([X, NEW], 1),
    names + ["ctr_item_bucket", "ctr_bucket_minus_item", "log_impr_item_bucket"])
