# Run from the repository root: .venv/bin/python experiments/lab/light_ctr.py
# LIGHT-USER ITEM CLICK RATES (2026-10-31): evaluation is dominated by light users (gap_profile: the long-gap AUC loss is a
# known-clicks effect). Is an item's click rate AMONG LIGHT USERS a different, better signal for light users than its overall
# click rate? Rates from fit days < 6 (labels of the fit period only), scored on day-6 events by known-clicks group.
import importlib.util
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments/13_retune_7days/retune.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
fit, test, _, _, n_items = e13.load_base(str(ROOT / "data/cache/13_base_c7_9c47a117c762.npz"))
_, hlen = e13.long_history(fit, fit, e13.HIST_LONG)
day = np.repeat(fit["day"], 10); h = np.repeat(hlen, 10).astype(int); it = fit["item"].astype(np.int64); y = fit["label"].astype(float)
GR = {"0": (0, 0), "1-5": (1, 5), "6-20": (6, 20), "21+": (21, 10**6)}
past = day < 6
def rate(m, K=20):
    c = np.bincount(it[m], y[m], minlength=n_items + 1); n = np.bincount(it[m], minlength=n_items + 1)
    g = y[m].mean(); return (c + K * g) / (n + K), n
r_all, n_all = rate(past)
rates = {k: rate(past & (h >= lo) & (h <= hi)) for k, (lo, hi) in GR.items()}
r_light, _ = rate(past & (h <= 5))
t = day == 6
print("day-6 AUC of the item click rate, by the test user's known clicks (columns: rate computed over all / over that group / over <=5)")
for k, (lo, hi) in GR.items():
    m = t & (h >= lo) & (h <= hi)
    print(f"  {k:5s} n_ev={m.sum() // 10:7,d}  all {e13.roc_auc(r_all[it[m]], y[m]):.4f}  group {e13.roc_auc(rates[k][0][it[m]], y[m]):.4f}  "
          f"light {e13.roc_auc(r_light[it[m]], y[m]):.4f}")
lr = np.log(r_light / r_all); print("corr(log rate all, log rate light) over items with 50+ light impressions:",
      np.corrcoef(np.log(r_all), np.log(r_light))[0, 1].round(4))
