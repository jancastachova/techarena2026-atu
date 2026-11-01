# Run from the repository root: .venv/bin/python experiments/lab/staleness.py
# STALENESS (2026-10-31): evaluation events run from day 7 to day 14 (median 7.7, 25% after 8.6) while click-rate statistics are
# frozen at day 7; every lab test spans one day. How fast does item popularity go stale? AUC on day d of the item click rate
# estimated from day d-k alone (k = 1..6), and from all days < d-k+1 (cumulative, frozen k-1 days before d). Train labels only.
import numpy as np, pandas as pd
T0, DAY = 1737331200, 86400
parts = []
for ch in pd.read_csv("data/train.csv", usecols=["item_id", "timestamp", "clicked"], chunksize=3_000_000):
    parts.append(pd.DataFrame({"item": ch.item_id.astype("category").cat.codes if False else ch.item_id, "day": (ch.timestamp - T0) // DAY, "y": ch.clicked}))
df = pd.concat(parts); df["day"] = df.day.astype(int)
print(df.groupby("day").y.agg(["size", "mean"]))
codes, uniq = pd.factorize(df["item"]); df["item"] = codes; n = len(uniq)
days = sorted(d for d in df.day.unique() if d >= 0)
C = {d: np.bincount(df.item[df.day == d], df.y[df.day == d], minlength=n) for d in days}
I = {d: np.bincount(df.item[df.day == d], minlength=n).astype(float) for d in days}
def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); p = y.sum(); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p))
for d in (5, 6):
    m = df.day == d; it, y = df.item[m].values, df.y[m].values
    out = []
    for k in range(1, d + 1):
        c, i = C[d - k], I[d - k]; s = (c + 1) / (i + 20)
        cc = sum(C[x] for x in days if x <= d - k); ii = sum(I[x] for x in days if x <= d - k); s2 = (cc + 1) / (ii + 20)
        out.append(f"k={k}: day-only {auc(s[it], y):.4f} cumul {auc(s2[it], y):.4f}")
    print(f"test day {d}: " + " | ".join(out), flush=True)
