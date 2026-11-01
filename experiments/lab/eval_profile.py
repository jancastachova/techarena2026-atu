# Run from the repository root: .venv/bin/python experiments/lab/eval_profile.py
# EVALUATION-SET PROFILE (2026-10-31): does the lab test (first event per user at/after day 6) look like evaluation (first event
# per user at/after day 7)? No labels are read from evaluation. Compares event timing, history length, history recency, history
# clicks AFTER the cutoff (clicks we may not have in train), and candidate-item freshness.
import json
import numpy as np, pandas as pd
T0, DAY = 1737331200, 86400
ev = pd.read_csv("data/evaluation.csv", usecols=["event_id", "user_id", "item_id", "timestamp", "history_item_ids", "history_timestamps"])
e1 = ev.drop_duplicates("event_id")
print(f"evaluation: {len(ev):,} rows, {len(e1):,} events, {e1.user_id.nunique():,} users")
d = (e1.timestamp.values - T0) / DAY
print("event day quantiles", np.round(np.quantile(d, [0, .1, .25, .5, .75, .9, .99, 1]), 3))
hts = [json.loads(s) for s in e1.history_timestamps]
L = np.array([len(h) for h in hts]); print("history length dist (0,1-5,6-19,20):", [(L == 0).mean().round(4), ((L >= 1) & (L <= 5)).mean().round(4), ((L >= 6) & (L < 20)).mean().round(4), (L == 20).mean().round(4)])
last = np.array([h[-1] if h else np.nan for h in hts]); gap = (e1.timestamp.values - last) / 3600
print("hours since last history click, quantiles", np.round(np.nanquantile(gap, [.1, .25, .5, .75, .9]), 2))
cut = T0 + 7 * DAY
post = np.array([sum(t >= cut for t in h) for h in hts]); print(f"history clicks at/after cutoff: {post.sum():,} in {(post > 0).mean():.4f} of events")
# lab analogue from train: first event per user at/after day 6 (days 6 only, before day 7)
tr = pd.read_csv("data/train.csv", usecols=["event_id", "user_id", "item_id", "timestamp", "history_timestamps"], chunksize=2_000_000)
items_before7, first6, seen_users = set(), {}, set()
items_before6 = set()
for ch in tr:
    t = ch.timestamp.values
    items_before7.update(ch.item_id.values[t < cut]); items_before6.update(ch.item_id.values[t < T0 + 6 * DAY])
    c6 = ch[(t >= T0 + 6 * DAY)].drop_duplicates("event_id")
    for r in c6.itertuples(index=False):
        if r.user_id not in first6 or r.timestamp < first6[r.user_id][0]:
            first6[r.user_id] = (r.timestamp, r.history_timestamps)
    print("chunk", flush=True) if False else None
f6 = list(first6.values()); d6 = np.array([x[0] for x in f6]); d6 = (d6 - T0) / DAY
print(f"lab (first event per user at/after day 6 in train): {len(f6):,} events; event day quantiles", np.round(np.quantile(d6, [0, .1, .25, .5, .75, .9, .99, 1]), 3))
h6 = [json.loads(x[1]) for x in f6]; L6 = np.array([len(h) for h in h6])
print("lab history length dist (0,1-5,6-19,20):", [(L6 == 0).mean().round(4), ((L6 >= 1) & (L6 <= 5)).mean().round(4), ((L6 >= 6) & (L6 < 20)).mean().round(4), (L6 == 20).mean().round(4)])
g6 = np.array([(x[0] - h[-1]) / 3600 if h else np.nan for x, h in zip(f6, h6)])
print("lab hours since last history click, quantiles", np.round(np.nanquantile(g6, [.1, .25, .5, .75, .9]), 2))
fresh = ~ev.item_id.isin(items_before7); print(f"eval candidate rows with item never in train before day 7: {fresh.mean():.4f}")
print(f"eval distinct items {ev.item_id.nunique():,}; in train-before-7 {ev.item_id.drop_duplicates().isin(items_before7).mean():.4f}")
print(f"eval users with any train row: {e1.user_id.isin(set(first6) | set()).mean():.4f} (users with a day>=6 train event)")
