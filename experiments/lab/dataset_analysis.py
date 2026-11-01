# Run from the repository root: .venv/bin/python experiments/lab/dataset_analysis.py
# Needs data/cache/dataset_raw.npz (experiments/lab/dataset_extract.py). Prints the numbers behind
# DATASET.md and writes its figures to docs/dataset/.
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path.cwd()
FIG = ROOT / "docs" / "dataset"
FIG.mkdir(parents=True, exist_ok=True)
z = np.load(ROOT / "data/cache/dataset_raw.npz")
T0, DAY = 1737331200, 86400
tr = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
te = {k[3:]: z[k] for k in z.files if k.startswith("te_")}
n_items, n_users = int(z["n_items"]), int(z["n_users"])
R = {}
def say(k, v):
    R[k] = v
    print(f"{k}: {v}", flush=True)
def save(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=110); plt.close()

E = len(tr["user"])
y = tr["label"].reshape(-1, 10)
ev_click = y.max(1)
items = tr["item"].reshape(-1, 10)
day = (tr["ts"] - T0) // DAY
say("train rows / events / users / items(all ids)", (len(tr["item"]), E, len(np.unique(tr["user"])), n_items))
say("events with a click", round(float(ev_click.mean()), 4))
say("clicks per event max", int(y.sum(1).max()))
say("events per day", np.bincount(day).tolist())
say("click share by day", [round(float(ev_click[day == d].mean()), 4) for d in range(7)])

# ---------------------------------------------------------------- positions and candidates
ctr_pos = y.mean(0)
say("CTR by position in file order", np.round(ctr_pos, 4).tolist())
dup = np.mean([len(set(r)) < 10 for r in items[:200000]])
say("events with a repeated candidate (first 200k)", float(dup))
plt.figure(figsize=(5, 3)); plt.bar(range(10), ctr_pos * 100); plt.xlabel("row position in event"); plt.ylabel("click rate %")
plt.title("Click rate by position"); save("ctr_by_position.png")

# ---------------------------------------------------------------- users
order = np.lexsort((tr["ts"], tr["user"]))
u_sorted, ts_sorted = tr["user"][order], tr["ts"][order]
n_ev_user = np.bincount(tr["user"], minlength=n_users)
nz = n_ev_user[n_ev_user > 0]
say("events per user: quantiles 10/50/90/99, max", (np.percentile(nz, [10, 50, 90, 99]).tolist(), int(nz.max())))
clk_user = np.bincount(tr["user"], ev_click, minlength=n_users)
m = n_ev_user >= 20
rate = clk_user[m] / n_ev_user[m]
p0 = ev_click.mean()
binom_var = np.mean(p0 * (1 - p0) / n_ev_user[m])
say("per-user click share (users with >=20 events): sd observed vs pure-chance sd",
    (round(float(rate.std()), 4), round(float(np.sqrt(binom_var)), 4)))
say("share of per-user variance that is real (not chance)", round(float(1 - binom_var / rate.var()), 3))
rng = np.random.default_rng(0)
sim = rng.binomial(n_ev_user[m], p0) / n_ev_user[m]
plt.figure(figsize=(5, 3)); plt.hist(rate, 60, alpha=0.6, label="observed", density=True)
plt.hist(sim, 60, alpha=0.6, label="if every user were alike", density=True)
plt.xlabel("share of the user's events with a click"); plt.legend(); plt.title("User click propensity (>= 20 events)")
save("user_propensity.png")
plt.figure(figsize=(5, 3)); plt.hist(np.log10(nz), 50); plt.xlabel("log10 events per user (train)"); plt.ylabel("users")
plt.title("Events per user"); save("events_per_user.png")

# ---------------------------------------------------------------- items
imp = np.bincount(tr["item"], minlength=n_items).astype(float)
clk = np.bincount(tr["item"], tr["label"], minlength=n_items).astype(float)
shown = imp > 0
say("items shown / clicked in train", (int(shown.sum()), int((clk > 0).sum())))
say("impressions per shown item: quantiles 1/10/50/90/99", np.percentile(imp[shown], [1, 10, 50, 90, 99]).tolist())
s = np.sort(imp[shown]); lor = np.cumsum(s) / s.sum()
say("impression Gini", round(float(1 - 2 * lor.mean()), 3))
mm = imp >= 200
ictr = clk[mm] / imp[mm]; pr = clk.sum() / imp.sum()
ib = np.mean(pr * (1 - pr) / imp[mm])
say("item CTR (items >=200 impressions): sd observed vs pure-chance sd", (round(float(ictr.std()), 4), round(float(np.sqrt(ib)), 4)))
say("corr(log impressions, CTR) items >=200 imp", round(float(np.corrcoef(np.log(imp[mm]), ictr)[0, 1]), 3))
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].hist(np.log10(imp[shown]), 60); ax[0].set_xlabel("log10 impressions per item"); ax[0].set_title("Exposure per item")
ax[1].scatter(np.log10(imp[mm]), ictr * 100, s=1, alpha=0.2); ax[1].set_xlabel("log10 impressions"); ax[1].set_ylabel("CTR %")
ax[1].set_title("Item CTR vs exposure"); save("items.png")

# ---------------------------------------------------------------- history bookkeeping between consecutive events
ev_o = order
same = u_sorted[1:] == u_sorted[:-1]
a, b = ev_o[:-1][same], ev_o[1:][same]
H, HT, HL = tr["hist"], tr["hist_ts"], tr["hlen"]
clicked_item = np.where(ev_click > 0, items[np.arange(E), y.argmax(1)], -1)
exact = 0; extra_clicks = []; extra_gap_frac = []; mism_kind = {"exact": 0, "click_missing": 0, "extra_unobserved": 0, "other": 0}
first_extra_ts_rel = []
for i, j in zip(a, b):
    prev = list(zip(H[i, :HL[i]], HT[i, :HL[i]]))
    if clicked_item[i] >= 0:
        prev = prev + [(clicked_item[i], tr["ts"][i])]
    nxt = list(zip(H[j, :HL[j]], HT[j, :HL[j]]))
    if nxt == prev[-20:] if len(prev) >= 20 else nxt == prev:
        mism_kind["exact"] += 1; continue
    # clicks in nxt that are newer than everything in prev
    last_prev = max([t for _, t in prev], default=-1)
    new = [(it, t) for it, t in nxt if t > last_prev]
    if clicked_item[i] >= 0 and (clicked_item[i], tr["ts"][i]) not in nxt and HL[j] < 20:
        mism_kind["click_missing"] += 1
    elif new:
        mism_kind["extra_unobserved"] += 1
        extra_clicks.append(len(new))
        first_extra_ts_rel.extend([(t - tr["ts"][i]) / (tr["ts"][j] - tr["ts"][i] + 1e-9) for _, t in new])
    else:
        mism_kind["other"] += 1
n_pairs = len(a)
say("consecutive event pairs (same user)", n_pairs)
say("next history = previous history + its click (exact)", {k: round(v / n_pairs, 4) for k, v in mism_kind.items()})
if extra_clicks:
    say("pairs with clicks between observed events: clicks per gap quantiles 50/90", np.percentile(extra_clicks, [50, 90]).tolist())

# ---------------------------------------------------------------- timing
gap = (ts_sorted[1:] - ts_sorted[:-1])[same]
say("gap between a user's consecutive events (minutes): quantiles 10/50/90", (np.percentile(gap, [10, 50, 90]) / 60).tolist())
say("consecutive events in the same minute", round(float(np.mean(gap == 0)), 4))
hour = (tr["ts"] % DAY) // 3600
plt.figure(figsize=(9, 3)); plt.subplot(1, 2, 1)
plt.bar(range(24), np.bincount(hour, minlength=24)); plt.xlabel("hour (UTC)"); plt.title("Events by hour")
plt.subplot(1, 2, 2); plt.hist(np.log10(gap / 60 + 1), 60); plt.xlabel("log10(1 + minutes to next event)"); plt.title("Gap between events")
save("timing.png")
say("click share by hour (UTC)", [round(float(ev_click[hour == h].mean()), 3) for h in range(24)])
tb_check = np.mean(((tr["ts"] % DAY) // 21600) == tr["tb"])
say("time_bucket matches timestamp", round(float(tb_check), 5))
# click share vs time since the last history click
last_click = np.where(HL > 0, HT[np.arange(E), np.maximum(HL - 1, 0)], 0)
since = np.where(HL > 0, (tr["ts"] - last_click) / 3600, np.nan)
bins = [0, 0.5, 2, 6, 24, 72, 168, 1e9]
bi = np.digitize(since, bins) - 1
say("click share by hours since last history click (0-.5,.5-2,2-6,6-24,24-72,72-168,168+)",
    [round(float(ev_click[bi == k].mean()), 4) for k in range(7)])

# ---------------------------------------------------------------- history content
say("history length quantiles 10/50/90 and share = 20", (np.percentile(HL, [10, 50, 90]).tolist(), round(float(np.mean(HL == 20)), 3)))
warm = np.mean((HT[HL > 0] < T0) & (H[HL > 0] >= 0))
say("history clicks from the warm-up period (share of all history entries, train events)", round(float(warm), 3))
rep_in_hist = np.mean([len(set(H[i, :HL[i]])) < HL[i] for i in range(0, E, 7)])
say("events whose history repeats an item", round(float(rep_in_hist), 3))
# candidate in history
in_hist = np.zeros((E, 10), bool)
for k in range(10):
    in_hist[:, k] = (H == items[:, k:k + 1]).any(1)
say("candidates already in the visible history", round(float(in_hist.mean()), 4))
say("CTR candidate in history vs not", (round(float(y[in_hist].mean()), 4), round(float(y[~in_hist].mean()), 4)))
# re-shown to the same user
prev_seen = 0; tot = 0
last_set = {}
for e in ev_o[:400000]:
    u = tr["user"][e]
    s_ = last_set.get(u)
    if s_ is not None:
        prev_seen += sum(it in s_ for it in items[e]); tot += 10
        s_.update(items[e].tolist())
    else:
        last_set[u] = set(items[e].tolist())
expected = None
say("candidates the user was shown in an earlier event (first 400k events by user)", round(prev_seen / max(tot, 1), 4))
# popularity-matched expectation of re-showing: sum over the user's earlier candidates of item exposure shares
share = imp / imp.sum()
say("expected re-show share if candidates were drawn by exposure share (rough, avg 13 earlier events x 10)",
    round(float(130 * (share ** 2).sum() / share.sum()), 4))

# ---------------------------------------------------------------- evaluation
EE = len(te["user"])
tday = (te["ts"] - T0) // DAY
say("evaluation rows / events / users", (len(te["item"]), EE, len(np.unique(te["user"]))))
say("evaluation events per day (7..13)", np.bincount(tday).tolist()[7:])
newu = ~np.isin(te["user"], tr["user"])
say("evaluation users without train events", round(float(newu.mean()), 4))
say("evaluation history length quantiles 10/50/90, empty share", (np.percentile(te["hlen"], [10, 50, 90]).tolist(), round(float(np.mean(te["hlen"] == 0)), 4)))
say("evaluation history clicks at/after the cutoff", int(((te["hist_ts"] >= T0 + 7 * DAY) & (te["hist"] >= 0)).sum()))
# per-user: train events of evaluation users, and gap from their last train event
last_tr = {}
for e in ev_o:
    last_tr[tr["user"][e]] = tr["ts"][e]
gap_ev = np.array([(te["ts"][i] - last_tr.get(te["user"][i], np.nan)) / 3600 if te["user"][i] in last_tr else np.nan for i in range(EE)])
say("evaluation: hours since the user's last train event, quantiles 10/50/90", np.nanpercentile(gap_ev, [10, 50, 90]).tolist())
# does the evaluation history extend beyond the user's last train event?
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].hist([np.minimum(tr["hlen"], 20), np.minimum(te["hlen"], 20)], bins=21, range=(0, 21), density=True, label=["train events", "evaluation events"])
ax[0].set_xlabel("visible history length"); ax[0].legend(); ax[0].set_title("History length")
ax[1].hist(np.log10(gap_ev[~np.isnan(gap_ev)] + 1), 60); ax[1].set_xlabel("log10(1 + hours since last train event)")
ax[1].set_title("Evaluation event timing"); save("evaluation.png")
te_imp = np.bincount(te["item"], minlength=n_items).astype(float)
say("corr of item exposure shares train vs evaluation", round(float(np.corrcoef(imp, te_imp)[0, 1]), 4))
json.dump({k: (v if not isinstance(v, np.generic) else v.item()) for k, v in R.items()},
          open(FIG / "numbers.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("figures in", FIG)

# ---------------------------------------------------------------- within-user: recency of the last click
# The click share falls with hours since the last history click; is that the user (heavy clickers
# always have a recent click) or the moment? Demean by each user's own click share (users with
# >= 20 events) and compare buckets of hours since the last click, and since the last event.
uu = tr["user"]
mu = clk_user / np.maximum(n_ev_user, 1)
keep = n_ev_user[uu] >= 20
dev = ev_click - mu[uu]
say("within-user click share deviation by hours since last history click (same buckets)",
    [round(float(dev[keep & (bi == k)].mean()), 4) for k in range(7)])
prev_ts = np.full(E, np.nan)
prev_ts[b] = tr["ts"][a]
since_ev = (tr["ts"] - prev_ts) / 3600
be = np.digitize(since_ev, bins) - 1
say("raw click share by hours since the previous EVENT",
    [round(float(ev_click[be == k].mean()), 4) for k in range(7)])
say("within-user deviation by hours since the previous EVENT",
    [round(float(dev[keep & (be == k)].mean()), 4) for k in range(7)])
say("share of events per bucket (since last click)", [round(float(np.mean(bi == k)), 3) for k in range(7)])
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
lab = ["<.5h", ".5-2h", "2-6h", "6-24h", "1-3d", "3-7d", ">7d"]
ax[0].bar(lab, [ev_click[bi == k].mean() for k in range(7)]); ax[0].set_ylim(0.5, 0.7)
ax[0].set_title("Events with a click, by time since last click"); ax[0].tick_params(axis="x", rotation=45)
ax[1].bar(lab, [dev[keep & (bi == k)].mean() for k in range(7)]); ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_title("Same, relative to the user's own rate"); ax[1].tick_params(axis="x", rotation=45)
save("recency_click_raw.png")      # DATASET.md shows recency_click.png: this vs the null of lab/dataset_recency_null.py
json.dump({k: (v if not isinstance(v, np.generic) else v.item()) for k, v in R.items()},
          open(FIG / "numbers.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
