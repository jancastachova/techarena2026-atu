# Run from the repository root: .venv/bin/python experiments/lab/dataset_reshow.py
# Needs data/cache/dataset_raw.npz. DATASET.md check on re-exposure: candidates the user was already
# shown in an earlier event (and did not click there, not in the visible history) vs fresh ones:
# share, click rate, click rate relative to the item's overall rate (popularity-adjusted), and
# how many events back the earlier showing was.
from pathlib import Path
import numpy as np
z = np.load(Path.cwd() / "data/cache/dataset_raw.npz")
ts, user, items = z["tr_ts"], z["tr_user"], z["tr_item"].reshape(-1, 10)
y = z["tr_label"].reshape(-1, 10); H = z["tr_hist"]
E = len(ts); n_items = int(z["n_items"])
imp = np.bincount(items.ravel(), minlength=n_items); clk = np.bincount(items.ravel(), y.ravel(), minlength=n_items)
irate = (clk + 20 * clk.sum() / imp.sum()) / (imp + 20)
o = np.lexsort((ts, user))
seen_at = {}                      # (user, item) -> index of the user's event where it was last shown
prev = np.zeros((E, 10), bool); back = np.full((E, 10), -1)
cnt = {}
for e in o:
    u = int(user[e]); k_ev = cnt.get(u, 0)
    for k in range(10):
        it = int(items[e, k])
        j = seen_at.get((u, it))
        if j is not None:
            prev[e, k] = True; back[e, k] = k_ev - j
        seen_at[(u, it)] = k_ev
    cnt[u] = k_ev + 1
inh = np.zeros((E, 10), bool)
for k in range(10):
    inh[:, k] = (H == items[:, k:k + 1]).any(1)
for name, m in [("re-shown, not in history", prev & ~inh), ("fresh, not in history", ~prev & ~inh),
                ("re-shown, in history", prev & inh), ("fresh, in history", ~prev & inh)]:
    exp = irate[items[m]].mean()
    print(f"{name:26s} {m.mean():7.2%} of rows  CTR {y[m].mean():.4f}  item-rate expectation {exp:.4f}  ratio {y[m].mean() / exp:.3f}")
b = back[prev & ~inh]
print("re-shown: events since the earlier showing, quantiles 25/50/75/90:", np.percentile(b, [25, 50, 75, 90]).tolist())
print("share of re-shown candidates that were shown in the immediately preceding event:", round(float(np.mean(b == 1)), 3))
