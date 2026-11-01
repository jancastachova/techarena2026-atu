# Run from the repository root: .venv/bin/python experiments/lab/dataset_recency_null.py
# Needs data/cache/dataset_raw.npz (lab/dataset_extract.py).
#
# DATASET.md check: within a user, is a click more likely the longer ago their last click was
# (fatigue / recovery), or is that the "hot hand" selection bias (events after a streak of non-clicks
# come from users whose own average those non-clicks pull down)? Null model: shuffle each user's
# click outcomes across their own events (keeps every user's rate, removes any time dependence),
# rebuild "hours since the last click" from the shuffled outcomes, and compare the within-user
# deviation per bucket with the observed one. The history's warm-up clicks are kept as they are
# for the first event; afterwards the last click time follows the (observed or shuffled) outcomes.
from pathlib import Path
import numpy as np
z = np.load(Path.cwd() / "data/cache/dataset_raw.npz")
ts, user, lab = z["tr_ts"], z["tr_user"], z["tr_label"].reshape(-1, 10).max(1).astype(float)
H, HT, HL = z["tr_hist"], z["tr_hist_ts"], z["tr_hlen"]
E = len(ts)
o = np.lexsort((ts, user)); us, tss = user[o], ts[o]
first = np.r_[True, us[1:] != us[:-1]]
warm_last = np.where(HL[o] > 0, HT[o, np.maximum(HL[o] - 1, 0)], np.nan)  # last visible click before the event
n = np.bincount(user); keep = n[us] >= 20
bins = [0, 0.5, 2, 6, 24, 72, 168, 1e9]
starts = np.flatnonzero(first); ends = np.r_[starts[1:], E]

def table(c):
    """c: click outcome per event in (user, ts) order -> mean within-user deviation per bucket."""
    last = np.empty(E)
    for s, e in zip(starts, ends):
        lc = warm_last[s]
        for k in range(s, e):
            last[k] = lc
            if c[k] > 0:
                lc = tss[k]
    since = (tss - last) / 3600
    b = np.digitize(since, bins) - 1
    mu = np.bincount(us, c, minlength=len(n)) / np.maximum(n, 1)
    dev = c - mu[us]
    return np.array([dev[keep & (b == k)].mean() for k in range(7)])

obs = table(lab[o])
rng = np.random.default_rng(0)
nulls = []
for r in range(5):
    c = lab[o].copy()
    for s, e in zip(starts, ends):
        c[s:e] = rng.permutation(c[s:e])
    nulls.append(table(c))
nulls = np.array(nulls)
print("buckets (hours since last click): <.5, .5-2, 2-6, 6-24, 24-72, 72-168, 168+")
print("observed  ", np.round(obs, 4))
print("null mean ", np.round(nulls.mean(0), 4), " sd", np.round(nulls.std(0), 4))
print("real effect (observed - null)", np.round(obs - nulls.mean(0), 4))
