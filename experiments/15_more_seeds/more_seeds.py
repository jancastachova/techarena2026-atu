#!/usr/bin/env python3
"""
TechArena CTR -- experiment 15: experiment 13 (standalone file, copied) with a different set of
MF seeds (15-29 instead of 0-14), so that its submission can be averaged with experiment 13's
(see average.py here): together 30 MF seeds and two LightGBM fits. Nothing else changes.

--- experiment 13's description (unchanged) ---------------------------------------------------
TechArena CTR -- experiment 13: experiment 12 (standalone file, copied) with the MF settings
re-tuned for the amount of data the final model trains on.

Until now every MF setting was tuned with an MF trained on 4 days (days 0-3 -> day 4), but the
submission MF trains on 7 days. Here --tune trains on days 0..C-2 and scores day C-1 of the
given cutoff's fit part, so with --cutoff 7 it compares settings on days 0-5 -> day 6 (still
fit-period data under the official split, never evaluation.csv).

TUNING (MF alone, days 0-5 -> day 6, AUC; everything else as experiment 12):
    training steps   2000    2500    3000    4000    5000 (12)  7000    9000
    1 seed           0.8114  0.8131  0.8137  0.8128  0.8107     0.8048  0.7982
    3 seeds                          0.8175          0.8148
    l2 0.03 / 0.08: 0.8093 / 0.8089; click-log weight 0.05 / 0.2: 0.8107 / 0.8067 (1 seed; kept)
Training stops earlier than before: even with 6 days of data, longer training memorises.
Changes vs experiment 12: MF_STEPS 5000 -> 3000, and N_SEEDS 10 -> 15 (about the same run
time as 10 x 5000 steps; averaging more seeds is a small, reliable gain).

--- experiment 12's description (unchanged) ---------------------------------------------------
TechArena CTR -- experiment 12: the full pipeline of experiment 11 in ONE self-contained file
(no imports from other experiment folders), plus an optional pre-training of the item vectors.

Everything an experiment needs is here: reading train.csv / evaluation.csv, the leak-free
"as-of" features, the time split, the matrix factorisation (MF), LightGBM, and the submission.
With --init-std 0.05 it reproduces experiment 11 exactly (same cached arrays, same MF and
LightGBM scores; see "Verification" below).

Pipeline (per cutoff day C; days counted from 2025-01-20 00:00 UTC)
  1. Base features (streaming train.csv in time order, then evaluation.csv in time order):
     - fit part = training events before C; test events = each user's FIRST event at/after C
       (validation, C < 7) or evaluation.csv (submission, C = 7); later events of test users
       are not used at all;
     - as-of features per candidate/event (repeat clicks, co-click similarity, transitions,
       item/user statistics, time of day ...): click histories are used as of each event;
       label statistics only change at day boundaries and are frozen at the cutoff;
     - a click log of every click known before the cutoff (history clicks, labelled clicks of
       fit events), and each event's timestamp.
  2. MF (separate process: torch and lightgbm crash when loaded into the same process on
     macOS). logit = g + b_user + b_item + mean(v_h over the user's known clicks) . v_item
     with ONE vector v per item; the user's known clicks are all logged clicks strictly
     before the event (most recent 100). Trained on impressions (binary cross-entropy) plus
     a click-log loss: each logged click vs 9 items drawn like candidates (in proportion to
     impressions), context = the user's OTHER clicks before the time limit. Averaged over 10
     seeds. Fit day d is scored by an MF trained on days < d (and clicks before day d); the
     test events by an MF trained on all fit days.
  3. LightGBM (binary objective, ranked by global AUC) on base + MF features + how each
     candidate compares with the other 9 of its event, fit on days 2..C-1. Trees are picked by
     early stopping on day C-1 (fit on days 2..C-2), then refit on days 2..C-1.

Change vs experiment 11: the item vectors start from random numbers with std 0.005 instead of
0.05 (--init-std; 0.05 reproduces experiment 11 exactly). Chosen inside the fit period only
(MF trained on days 0-3 + clicks before day 4, AUC on fit day 4):
                                                   1 seed   3 seeds
    random start, std 0.05 (= experiment 11)       0.7838   0.7946
    random start, std 0.01 / 0.005 / 0.002           -      0.7969 / 0.7973 / 0.7975
    co-click pre-training (PPMI-SVD), scale 0.02   0.7915   0.7944
    co-click pre-training, scale 0.1/0.05/0.01/0.005: 0.7874 / 0.7893 / 0.7926 / 0.7933 (1 seed)
Co-click pre-training (--pretrain ppmi, kept as an option) looked good with 1 seed but gave
nothing with 3: every seed starts from the same vectors, so it only removes seed noise, which
averaging already does. A small random start keeps its gain under averaging.

Optional pre-training (--pretrain ppmi): before training, the item vectors are initialised
from co-click statistics instead of random numbers: a positive-PMI matrix of "clicked by the
same user" counts (clicks before the time limit only), reduced with a truncated SVD.

Usage
    # submission (official cutoff)
    .venv/bin/python experiments/13_retune_7days/retune.py --cutoff 7 \
        --train data/train.csv --evaluation data/evaluation.csv --out submissions/13_retune_7days.csv
    # validation (cutoff day 5): AUC overall and per segment, nothing written
    .venv/bin/python experiments/12_standalone/standalone.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv
    # MF variants, fit period only (MF trained on days 0-3, AUC on fit day 4)
    .venv/bin/python experiments/12_standalone/standalone.py --cutoff 5 \
        --train data/train.csv --evaluation data/evaluation.csv --tune none,ppmi --tune-seeds 1

Verification (exact reproduction of experiment 11)
    --verify-base <npz>     compare the cached base features with another base cache
    --use-mf <npz>          use precomputed MF scores (mf_fit / mf_test) instead of training
"""
import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

TRAIN_START = 1737331200      # 2025-01-20 00:00 UTC
DAY = 86400
HOUR = 3600.0
OFFICIAL_CUTOFF = 7
EV_SIZE = 10
HIST_MAX = 20                 # length of history_item_ids (stored per event for the base features)
FIRST_MODEL_DAY = 2           # fit days >= 2 get MF scores and are used by LightGBM

# as-of base features
CTR_PRIOR = 0.063             # approx. training-wide click rate per row, used before stats exist
EVENT_CLICK_PRIOR = 0.63      # approx. share of events that contain a click
SMOOTHING_K = 50              # pseudo-count for item CTR shrinkage
USER_SMOOTHING_K = 5          # pseudo-count (in events) for user click-rate shrinkage
CO_WINDOW = 20                # a new click is paired with the user's previous CO_WINDOW clicks
TAU_HIST = 24 * HOUR          # decay of history clicks when weighting similarity
TAU_POP = 24 * HOUR           # decay of item popularity ("trending")
TAU_REPEAT = 24 * HOUR        # decay of a repeat click on the candidate
SIM_SCALE = 100.0             # similarities enter as log1p(SIM_SCALE * s) (they are heavy-tailed)

# MF
MF_DIM = 32
MF_L2 = 5e-2
MF_STEPS = 3000              # experiment 12: 5000 (see TUNING)
MF_BATCH_EVENTS = 820         # ~8,200 rows per step
MF_LR = 3e-3
HIST_LONG = 100               # user vector over the most recent HIST_LONG known clicks
AUX_WEIGHT = 0.1              # weight of the click-log loss
N_NEG = 9                     # sampled negatives per logged click
N_SEEDS = 15                # experiment 12: 10
SEED_OFFSET = 15            # seeds SEED_OFFSET .. SEED_OFFSET + N_SEEDS - 1 (experiment 13: 0)
PRETRAIN = "none"             # "none" (= experiment 11) or "ppmi" (see --tune results)
INIT_STD_DEFAULT = 0.005      # std of the random initial item vectors (experiments 04-11: 0.05); see --init-std
INIT_STD = INIT_STD_DEFAULT

LGB_PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=0, deterministic=True, force_row_wise=True,
)

CAND_FEATURES = [
    "rep_any", "rep_count", "rep_recency",                       # candidate already in the history
    "sim_sum", "sim_decay", "sim_last", "sim_max",               # co-click similarity (log-scaled)
    "trans_decay", "trans_last",                                 # "clicked A, next B" transitions
    "item_ctr_logit", "item_log_impr", "item_log_pop_decay", "item_log_clicks",
]
EVENT_FEATURES = [
    "hist_len", "no_hist", "log_hours_since_last", "clicks_24h", "user_rate_logit", "user_log_events",
    "tb_1", "tb_2", "tb_3", "day_type",
    "max_sim_sum", "mean_sim_sum", "n_rep", "max_ctr", "mean_ctr",
]
NC, NE = len(CAND_FEATURES), len(EVENT_FEATURES)
MF_FEATURES = ["mf_logit", "mf_item_bias", "mf_user_bias", "mf_hist_dot"]
REL_FEATURES = ["mf_logit_minus_max", "mf_logit_rank", "mf_logit_share", "mf_event_lse",
                "sim_sum_minus_max", "sim_sum_rank", "ctr_minus_max"]
BASE_FEATURES = CAND_FEATURES + EVENT_FEATURES


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def roc_auc(scores, labels):
    """Global ROC AUC, ties counted as one half (same definition as the leaderboard)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def code_hash(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode() if isinstance(p, str) else p)
    return h.hexdigest()[:12]


# =================================================================== as-of base features

def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class State:
    """Everything learned so far from the stream (only past information)."""

    def __init__(self):
        self.item_impr = defaultdict(int)
        self.item_clk = defaultdict(int)
        self.user_events = defaultdict(int)
        self.user_events_clk = defaultdict(int)
        self.total_rows = 0
        self.total_clk = 0
        self.user_log = {}                       # user -> list of (ts, item), ascending, recent tail kept
        self.item_n = defaultdict(int)           # clicks per item seen in the click log
        self.pop_val = {}                        # item -> decayed click count at pop_t
        self.pop_t = {}
        self.co = defaultdict(lambda: defaultdict(int))     # item -> item -> co-click count
        self.trans = defaultdict(lambda: defaultdict(int))  # a -> b: user clicked b right after a
        self.trans_out = defaultdict(int)
        self.pending = []                        # labels waiting for the next day boundary

    def add_history(self, user, hist_items, hist_ts):
        """Merge the clicks of an event's history into the user's click log."""
        log_ = self.user_log.get(user)
        if log_ is None:
            log_ = []
            self.user_log[user] = log_
        last_ts = log_[-1][0] if log_ else -1
        same_ts = {it for t, it in log_ if t == last_ts} if log_ else set()
        seen_now = set()
        for it, t in zip(hist_items, hist_ts):
            if t < last_ts or (t == last_ts and it in same_ts) or (t, it) in seen_now:
                continue
            seen_now.add((t, it))
            self._add_click(log_, it, t)

    def _add_click(self, log_, b, tb):
        self.item_n[b] += 1
        if b in self.pop_t:                      # clicks can arrive slightly out of time order
            pt = self.pop_t[b]
            if tb >= pt:
                self.pop_val[b] = self.pop_val[b] * math.exp(-(tb - pt) / TAU_POP) + 1.0
                self.pop_t[b] = tb
            else:
                self.pop_val[b] += math.exp(-(pt - tb) / TAU_POP)
        else:
            self.pop_val[b] = 1.0
            self.pop_t[b] = tb
        co_b = self.co[b]
        for _, a in log_[-CO_WINDOW:]:
            if a != b:
                co_b[a] += 1
                self.co[a][b] += 1
        if log_:
            a = log_[-1][1]
            self.trans[a][b] += 1
            self.trans_out[a] += 1
        log_.append((tb, b))
        if len(log_) > 2 * CO_WINDOW:
            del log_[:-CO_WINDOW]

    def add_labels(self, user, items, labels):
        """Queued; label statistics only change at flush() (day boundaries in the fit part)."""
        self.pending.append((user, items, labels))

    def flush(self):
        for user, items, labels in self.pending:
            clicked = 0
            for it, y in zip(items, labels):
                self.item_impr[it] += 1
                self.item_clk[it] += y
                self.total_rows += 1
                self.total_clk += y
                clicked |= y
            self.user_events[user] += 1
            self.user_events_clk[user] += clicked
        self.pending = []

    def event_features(self, user, ts, time_bucket, day_type, hist_items, hist_ts, items):
        """Returns (candidate feature rows, event feature row) using only past info."""
        g_ctr = self.total_clk / self.total_rows if self.total_rows else CTR_PRIOR
        hist = []
        last_seen = {}
        hist_count = defaultdict(int)
        for it, t in zip(hist_items, hist_ts):
            w = math.exp(-max(0.0, ts - t) / TAU_HIST)
            n_h = self.item_n.get(it, 0)
            hist.append((self.co.get(it), math.sqrt(n_h) if n_h else 0.0, w,
                         self.trans.get(it), self.trans_out.get(it, 0)))
            hist_count[it] += 1
            last_seen[it] = t
        last_entry = hist[-1] if hist else None

        cand = []
        raw_sim_sums = []
        for c in items:
            cnt = hist_count.get(c, 0)
            rep_rec = math.exp(-max(0.0, ts - last_seen[c]) / TAU_REPEAT) if cnt else 0.0
            n_c = self.item_n.get(c, 0)
            sim_sum = sim_dec = sim_max = sim_last = 0.0
            trans_dec = trans_last = 0.0
            if hist:
                sq_c = math.sqrt(n_c) if n_c else 0.0
                for co_h, sq_h, w, tr_h, tr_out in hist:
                    if sq_c and co_h is not None:
                        v = co_h.get(c)
                        if v:
                            s = v / (sq_h * sq_c)
                            sim_sum += s
                            sim_dec += w * s
                            if s > sim_max:
                                sim_max = s
                    if tr_h is not None:
                        v = tr_h.get(c)
                        if v:
                            trans_dec += w * v / tr_out
                co_l, sq_l, _, tr_l, tro_l = last_entry
                if sq_c and co_l is not None:
                    v = co_l.get(c)
                    if v:
                        sim_last = v / (sq_l * sq_c)
                if tr_l is not None:
                    v = tr_l.get(c)
                    if v:
                        trans_last = v / tro_l
            imp = self.item_impr.get(c, 0)
            ctr = (self.item_clk.get(c, 0) + SMOOTHING_K * g_ctr) / (imp + SMOOTHING_K)
            if c in self.pop_t:
                pop = self.pop_val[c] * math.exp(-max(0.0, ts - self.pop_t[c]) / TAU_POP)
            else:
                pop = 0.0
            raw_sim_sums.append(sim_sum)
            cand.append([
                1.0 if cnt else 0.0, float(cnt), rep_rec,
                math.log1p(SIM_SCALE * sim_sum), math.log1p(SIM_SCALE * sim_dec),
                math.log1p(SIM_SCALE * sim_last), math.log1p(SIM_SCALE * sim_max),
                math.log1p(10.0 * trans_dec), math.log1p(10.0 * trans_last),
                logit(ctr), math.log1p(imp), math.log1p(pop), math.log1p(n_c),
            ])

        n_hist = len(hist_items)
        if n_hist:
            log_since_last = math.log1p(max(0.0, (ts - hist_ts[-1]) / HOUR))
            clicks_24h = sum(1 for t in hist_ts if ts - t <= DAY)
        else:
            log_since_last = 0.0
            clicks_24h = 0
        ue = self.user_events.get(user, 0)
        uc = self.user_events_clk.get(user, 0)
        user_rate = (uc + USER_SMOOTHING_K * EVENT_CLICK_PRIOR) / (ue + USER_SMOOTHING_K)
        i_sim = CAND_FEATURES.index("sim_sum")
        i_ctr = CAND_FEATURES.index("item_ctr_logit")
        ev = [
            n_hist / 20.0, 1.0 if n_hist == 0 else 0.0, log_since_last, clicks_24h / 20.0,
            logit(user_rate), math.log1p(ue),
            1.0 if time_bucket == 1 else 0.0, 1.0 if time_bucket == 2 else 0.0,
            1.0 if time_bucket == 3 else 0.0, float(day_type),
            max(r[i_sim] for r in cand), math.log1p(SIM_SCALE * sum(raw_sim_sums) / len(cand)),
            sum(r[0] for r in cand), max(r[i_ctr] for r in cand), sum(r[i_ctr] for r in cand) / len(cand),
        ]
        return cand, ev


def parse_event(rows, ix):
    first = rows[0]
    return (first[ix["user_id"]], int(first[ix["timestamp"]]), int(first[ix["time_bucket"]]),
            int(first[ix["day_type"]]), json.loads(first[ix["history_item_ids"]]),
            [int(t) for t in json.loads(first[ix["history_timestamps"]])],
            [r[ix["item_id"]] for r in rows])


def iter_train_events(path):
    """Yields (header index, rows) per event; train.csv keeps events contiguous."""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        ix = {c: i for i, c in enumerate(next(r))}
        cur, rows, done = None, [], set()
        for row in r:
            ev = row[ix["event_id"]]
            if ev != cur:
                if rows:
                    yield ix, rows
                if ev in done:
                    raise ValueError(f"event {ev} is not contiguous in {path}")
                done.add(ev)
                cur, rows = ev, []
            rows.append(row)
        if rows:
            yield ix, rows


class Split:
    """Columns of one split (fit or test); rows are grouped by event, EV_SIZE per event."""

    def __init__(self):
        self.cand, self.ev = array("f"), array("f")
        self.user, self.item, self.hist = array("i"), array("i"), array("i")
        self.hlen, self.label, self.day = array("b"), array("b"), array("b")
        self.ts = array("q")
        self.rowid = []

    def add(self, cand, ev, user, ts, items, h_items, enc_u, enc_i):
        if len(items) != EV_SIZE:
            raise ValueError(f"event with {len(items)} candidates, expected {EV_SIZE}")
        for r in cand:
            self.cand.extend(r)
        self.ev.extend(ev)
        self.user.append(enc_u(user))
        self.item.extend(enc_i(it) for it in items)
        codes = [enc_i(it) for it in h_items[-HIST_MAX:]]
        self.hist.extend(codes + [0] * (HIST_MAX - len(codes)))
        self.hlen.append(len(codes))
        self.ts.append(ts)

    def to_numpy(self):
        n_ev = len(self.user)
        out = dict(
            cand=np.frombuffer(self.cand, dtype=np.float32).reshape(n_ev * EV_SIZE, NC),
            ev=np.frombuffer(self.ev, dtype=np.float32).reshape(n_ev, NE),
            user=np.frombuffer(self.user, dtype=np.int32),
            item=np.frombuffer(self.item, dtype=np.int32),
            hist=np.frombuffer(self.hist, dtype=np.int32).reshape(n_ev, HIST_MAX),
            hlen=np.frombuffer(self.hlen, dtype=np.int8),
            day=np.frombuffer(self.day, dtype=np.int8),
            ts=np.frombuffer(self.ts, dtype=np.int64),
        )
        if len(self.label):
            out["label"] = np.frombuffer(self.label, dtype=np.int8)
        if self.rowid:
            out["rowid"] = np.array(self.rowid)
        return {k: v.copy() for k, v in out.items()}


def build(train_path, eval_path, cutoff):
    """Streams the data once. Returns (fit, test, eval_order, n_users, n_items); fit also holds
    the click log (clk_user, clk_item, clk_ts) of every click known before the cutoff."""
    st = State()
    users, items = {}, {}
    clicks = set()

    def enc_u(u):
        c = users.get(u)
        if c is None:
            c = users[u] = len(users)
        return c

    def enc_i(i):                    # item codes start at 1, 0 is the history padding
        c = items.get(i)
        if c is None:
            c = items[i] = len(items) + 1
        return c

    def log_history(user, h_items, h_ts):
        uc = enc_u(user)
        for it, t in zip(h_items, h_ts):
            clicks.add((uc, enc_i(it), t))

    fit, test = Split(), Split()
    test_users = set()
    n_skipped = 0
    prev_ts, cur_day = -1, 0
    t0 = time.time()
    log(f"base features with cutoff day {cutoff}: streaming train.csv ...")
    for n_ev, (ix, rows) in enumerate(iter_train_events(train_path)):
        user, ts, tb, dt, h_items, h_ts, its = parse_event(rows, ix)
        if ts < prev_ts:
            raise ValueError("train.csv is not sorted by timestamp")
        prev_ts = ts
        day = (ts - TRAIN_START) // DAY
        if not 0 <= day <= 6:
            raise ValueError(f"unexpected training timestamp {ts}")
        if day != cur_day:
            st.flush()               # labels of finished fit days become visible
            cur_day = day
        ys = [int(r[ix["clicked"]]) for r in rows]
        if sum(ys) > 1:
            raise ValueError("event with more than one click")
        if day < cutoff:
            st.add_history(user, h_items, h_ts)
            cand, ev = st.event_features(user, ts, tb, dt, h_items, h_ts, its)
            fit.add(cand, ev, user, ts, its, h_items, enc_u, enc_i)
            fit.label.extend(ys)
            fit.day.append(day)
            st.add_labels(user, its, ys)
            log_history(user, h_items, h_ts)
            for it, y in zip(its, ys):
                if y:
                    clicks.add((enc_u(user), enc_i(it), ts))
        elif user not in test_users:  # first event of this user at/after the cutoff
            test_users.add(user)
            st.add_history(user, h_items, h_ts)
            cand, ev = st.event_features(user, ts, tb, dt, h_items, h_ts, its)
            test.add(cand, ev, user, ts, its, h_items, enc_u, enc_i)
            test.label.extend(ys)
            test.day.append(day)
            test.rowid.extend(r[ix["rowid"]] for r in rows)
            log_history(user, h_items, h_ts)
        else:
            n_skipped += 1            # later events of test users: not used at all
        if (n_ev + 1) % 200_000 == 0:
            log(f"  {n_ev+1:,} events ({time.time()-t0:.0f}s)")
    if st.pending and cutoff < OFFICIAL_CUTOFF:
        raise AssertionError("labels after the cutoff were queued")

    eval_order = np.array([], dtype=str)
    if cutoff == OFFICIAL_CUTOFF:
        st.flush()
        log("base features: evaluation.csv in time order ...")
        with open(eval_path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            ix = {c: i for i, c in enumerate(next(r))}
            eval_rows = list(r)
        events = defaultdict(list)
        for i, row in enumerate(eval_rows):
            events[row[ix["event_id"]]].append(i)
        order = sorted(events.values(), key=lambda idx: int(eval_rows[idx[0]][ix["timestamp"]]))
        for idx in order:
            rows = [eval_rows[i] for i in idx]
            user, ts, tb, dt, h_items, h_ts, its = parse_event(rows, ix)
            if user in test_users:
                raise ValueError("evaluation.csv has more than one event for a user")
            test_users.add(user)
            st.add_history(user, h_items, h_ts)
            cand, ev = st.event_features(user, ts, tb, dt, h_items, h_ts, its)
            test.add(cand, ev, user, ts, its, h_items, enc_u, enc_i)
            test.day.append(min((ts - TRAIN_START) // DAY, 127))
            test.rowid.extend(r[ix["rowid"]] for r in rows)
            log_history(user, h_items, h_ts)
        eval_order = np.array([row[ix["rowid"]] for row in eval_rows])

    cutoff_ts = TRAIN_START + cutoff * DAY
    clk = np.array(sorted(clicks), dtype=np.int64).reshape(-1, 3)
    if len(clk) and clk[:, 2].max() >= cutoff_ts:
        raise AssertionError("click log contains a click at/after the cutoff")
    fit_np, test_np = fit.to_numpy(), test.to_numpy()
    fit_np["clk_user"], fit_np["clk_item"], fit_np["clk_ts"] = clk[:, 0], clk[:, 1], clk[:, 2]

    # every clicked fit candidate is in the click log at (user, item, event timestamp)
    lab = fit_np["label"].reshape(-1, EV_SIZE)
    ev_i, pos = np.nonzero(lab)
    clicked = set(zip(fit_np["user"][ev_i].tolist(), fit_np["item"].reshape(-1, EV_SIZE)[ev_i, pos].tolist(),
                      fit_np["ts"][ev_i].tolist()))
    logged = set(zip(clk[:, 0].tolist(), clk[:, 1].tolist(), clk[:, 2].tolist()))
    if not clicked <= logged:
        raise AssertionError(f"{len(clicked - logged)} clicked candidates missing from the click log")
    log(f"base features done ({time.time()-t0:.0f}s): {len(fit.user):,} fit events, {len(test.user):,} "
        f"test events, {n_skipped:,} later events of test users skipped, {len(clk):,} logged clicks")
    return fit_np, test_np, eval_order, len(users), len(items) + 1


def load_base(path):
    z = np.load(path)
    fit = {k[3:]: z[k] for k in z.files if k.startswith("tr_")}
    test = {k[3:]: z[k] for k in z.files if k.startswith("ev_")}
    return fit, test, z["eval_order"], int(z["n_users"]), int(z["n_items"])


# =================================================================== user histories from the click log

def long_history(d, fit, k):
    """Per event of d: the user's logged clicks strictly before the event, newest first, at
    most k (0 = padding), and their count. The click log is fit["clk_*"]."""
    u, it, ts = fit["clk_user"].astype(np.int64), fit["clk_item"], fit["clk_ts"].astype(np.int64)
    order = np.lexsort((it, ts, u))
    key = u[order] * 10_000_000_000 + ts[order]
    it = it[order]
    eu = d["user"].astype(np.int64)
    start = np.searchsorted(key, eu * 10_000_000_000, side="left")
    end = np.searchsorted(key, eu * 10_000_000_000 + d["ts"].astype(np.int64), side="left")
    hist = np.zeros((len(eu), k), dtype=np.int32)
    for j in range(k):
        idx = end - 1 - j
        ok = idx >= start
        hist[ok, j] = it[idx[ok]]
    return hist, np.minimum(end - start, k).astype(np.int8 if k < 128 else np.int16)


def check_against_event_history(d, fit, name):
    """The 20 most recent logged clicks must be the event's own history (as a multiset)."""
    h20, n20 = long_history(d, fit, 20)
    same = sum(np.array_equal(np.sort(h20[e, :n20[e]]), np.sort(d["hist"][e, :d["hlen"][e]]))
               for e in range(len(n20)))
    rate = same / len(n20)
    log(f"  {name}: last 20 logged clicks == event history for {rate:.4%} of events")
    if rate < 0.99:
        raise AssertionError(f"{name}: click log and event histories disagree ({rate:.2%})")


def with_long_history(d, fit, k):
    d = dict(d)
    d["hist"], d["hlen"] = long_history(d, fit, k)
    return d


def other_clicks_examples(fit, day, k):
    """Click-log examples from clicks before `day`: context = the user's OTHER clicks before
    `day` (most recent k, the click itself excluded), target = the clicked item."""
    u, it, ts = fit["clk_user"], fit["clk_item"], fit["clk_ts"]
    keep = ts < TRAIN_START + day * DAY
    u, it, ts = u[keep], it[keep], ts[keep]
    o = np.lexsort((it, ts, u))
    u, it, ts = u[o], it[o], ts[o]
    bounds = np.r_[0, np.flatnonzero(np.diff(u)) + 1, len(u)]
    ctx = np.zeros((len(u), k), dtype=np.int64)
    ln = np.zeros(len(u), dtype=np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        items = it[a:b]
        for j in range(b - a):
            others = np.r_[items[:j], items[j + 1:]][-k:]
            ctx[a + j, :len(others)] = others
            ln[a + j] = len(others)
    has = ln > 0
    return ctx[has], ln[has], it[has], ts[has]


def ppmi_svd_vectors(fit, day, n_items, dim, scale=0.05):
    """Item vectors from co-click statistics of clicks before `day`: positive PMI of 'clicked by
    the same user', truncated SVD, U * sqrt(S), rescaled to standard deviation `scale`. Items
    without clicks get no vector (NaN rows; the caller keeps their random init)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    keep = fit["clk_ts"] < TRAIN_START + day * DAY
    pairs = np.unique(np.stack([fit["clk_user"][keep], fit["clk_item"][keep]], 1), axis=0)
    n_users = int(pairs[:, 0].max()) + 1
    A = csr_matrix((np.ones(len(pairs), dtype=np.float64), (pairs[:, 0], pairs[:, 1])),
                   shape=(n_users, n_items))
    C = (A.T @ A).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    total = C.sum()
    marg = np.asarray(C.sum(1)).ravel()
    C = C.tocoo()
    pmi = np.log(C.data * total / (marg[C.row] * marg[C.col]))
    pos = pmi > 0
    P = csr_matrix((pmi[pos].astype(np.float32), (C.row[pos], C.col[pos])), shape=(n_items, n_items))
    U, S, _ = svds(P, k=dim, random_state=0)
    V = U * np.sqrt(S)[None, :]
    has = marg > 0
    V = V / V[has].std() * scale
    V[~has] = np.nan
    return V.astype(np.float32)


# =================================================================== MF (torch; own process)

def mf_stage(base_file, out_file, cutoff, pretrain, tune=None, tune_seeds=1):
    """Trains the MFs of the time split and writes mf_fit / mf_test, or (tune) compares MF
    variants on fit day 4 (trained on days 0-3)."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)

    class TiedMF(torch.nn.Module):
        """logit = g + b_user + b_item + mean(v_h for h in history) . v_item, one v per item."""

        def __init__(self, n_users, n_items, dim):
            super().__init__()
            self.g = torch.nn.Parameter(torch.tensor(math.log(CTR_PRIOR / (1 - CTR_PRIOR))))
            self.bu = torch.nn.Embedding(n_users, 1)
            self.bi = torch.nn.Embedding(n_items, 1)
            qi = torch.nn.Embedding(n_items, dim)       # created as in experiment 04 (same random
            self.yh = torch.nn.Embedding(n_items, dim, padding_idx=0)   # numbers), then tied
            torch.nn.init.zeros_(self.bu.weight)
            torch.nn.init.zeros_(self.bi.weight)
            torch.nn.init.normal_(qi.weight, std=INIT_STD)
            torch.nn.init.normal_(self.yh.weight, std=INIT_STD)
            with torch.no_grad():
                self.yh.weight[0].zero_()
            self.qi = self.yh                           # candidates are never item 0

        def forward(self, u, items, hist, hlen):
            hv = self.yh(hist).sum(1) / hlen.clamp(min=1).unsqueeze(1)
            q = self.qi(items)
            bu = self.bu(u).expand(-1, items.shape[1])
            bi = self.bi(items).squeeze(2)
            dot = (hv.unsqueeze(1) * q).sum(2)
            reg = (q ** 2).sum() / items.shape[1] + (hv ** 2).sum()
            return self.g + bu + bi + dot, bi, bu, dot, reg

    def event_tensors(d, ev):
        return (torch.from_numpy(d["user"][ev].astype(np.int64)),
                torch.from_numpy(d["item"].reshape(-1, EV_SIZE)[ev].astype(np.int64)),
                torch.from_numpy(d["hist"][ev].astype(np.int64)),
                torch.from_numpy(d["hlen"][ev].astype(np.float32)))

    def train(d, events, clicks, n_users, n_items, seed, init_vectors=None):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = TiedMF(n_users, n_items, MF_DIM)
        if init_vectors is not None:
            with torch.no_grad():
                ok = ~np.isnan(init_vectors).any(1)
                ok[0] = False                               # padding row stays zero
                w = model.yh.weight.detach().numpy()        # shares memory with the parameter
                w[ok] = init_vectors[ok]
        opt = torch.optim.Adam(model.parameters(), lr=MF_LR)
        Y = torch.from_numpy(d["label"].reshape(-1, EV_SIZE).astype(np.float32))
        rows = (events[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
        p = np.bincount(d["item"][rows], minlength=n_items).astype(np.float64)
        p /= p.sum()
        ctx, ctx_len, target, _ = clicks
        ctx, ctx_len, target = torch.from_numpy(ctx), torch.from_numpy(ctx_len), torch.from_numpy(target)
        steps, t0 = 0, time.time()
        while steps < MF_STEPS:
            perm = rng.permutation(events)
            for s in range(0, len(perm), MF_BATCH_EVENTS):
                ev = perm[s:s + MF_BATCH_EVENTS]
                logit_, _, _, _, reg = model(*event_tensors(d, ev))
                loss = F.binary_cross_entropy_with_logits(logit_, Y[torch.from_numpy(ev)])
                total = loss + MF_L2 * reg / logit_.numel()
                if AUX_WEIGHT and len(target):
                    k = torch.from_numpy(rng.integers(0, len(target), size=len(ev)))
                    neg = torch.from_numpy(rng.choice(n_items, size=(len(ev), N_NEG), p=p))
                    pos_item = target[k].unsqueeze(1)
                    cand = torch.cat([pos_item, neg], 1)
                    hv = model.yh(ctx[k]).sum(1) / ctx_len[k].clamp(min=1).unsqueeze(1)
                    q = model.qi(cand)
                    a_logit = model.bi(cand).squeeze(2) + (hv.unsqueeze(1) * q).sum(2)
                    same = torch.cat([torch.zeros_like(pos_item, dtype=torch.bool), neg == pos_item], 1)
                    a_logit = a_logit.masked_fill(same, -1e9)
                    a_loss = F.cross_entropy(a_logit, torch.zeros(len(ev), dtype=torch.long))
                    a_reg = (q ** 2).sum() / cand.shape[1] + (hv ** 2).sum()
                    total = total + AUX_WEIGHT * (a_loss + MF_L2 * a_reg / a_logit.numel())
                opt.zero_grad()
                total.backward()
                opt.step()
                steps += 1
                if steps >= MF_STEPS:
                    break
        log(f"    seed {seed}: {MF_STEPS} steps on {len(events):,} events + {len(target):,} "
            f"logged clicks ({time.time()-t0:.0f}s)")
        return model

    @torch.no_grad()
    def score(model, d, events):
        model.eval()
        out = []
        for s in range(0, len(events), 4096):
            ev = events[s:s + 4096]
            logit_, bi, bu, dot, _ = model(*event_tensors(d, ev))
            out.append(torch.stack([logit_, bi, bu, dot], 2).reshape(-1, len(MF_FEATURES)).numpy())
        return np.concatenate(out).astype(np.float32)

    def averaged(d, events, clicks, n_users, n_items, n_seeds, init_vectors, score_fn):
        out = None
        for seed in range(SEED_OFFSET, SEED_OFFSET + n_seeds):
            s = score_fn(train(d, events, clicks, n_users, n_items, seed, init_vectors))
            out = s if out is None else out + s
        return out / n_seeds

    def init_for(kind, fit, day, n_items):
        global INIT_STD
        INIT_STD = INIT_STD_DEFAULT
        if kind == "none":
            return None
        if kind.startswith("std:"):                        # random init with another scale
            INIT_STD = float(kind.split(":")[1])
            return None
        if kind.startswith("ppmi"):
            scale = float(kind.split(":")[1]) if ":" in kind else 0.05
            t0 = time.time()
            V = ppmi_svd_vectors(fit, day, n_items, MF_DIM, scale)
            log(f"    PPMI-SVD item vectors for clicks before day {day} ({time.time()-t0:.0f}s)")
            return V
        raise ValueError(f"unknown pretraining {kind!r}")

    fit, test, _, n_users, n_items = load_base(base_file)
    check_against_event_history(fit, fit, "fit events")
    check_against_event_history(test, fit, "test events")
    fit_l, test_l = with_long_history(fit, fit, HIST_LONG), with_long_history(test, fit, HIST_LONG)
    ev_day = fit["day"]

    if tune:
        global MF_STEPS, MF_L2, AUX_WEIGHT, INIT_STD
        defaults = (MF_STEPS, MF_L2, AUX_WEIGHT, INIT_STD_DEFAULT)
        day = cutoff - 1
        train_ev, target = np.flatnonzero(ev_day < day), np.flatnonzero(ev_day == day)
        rows = (target[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
        clk = other_clicks_examples(fit, day, HIST_LONG)
        for kind in tune:
            MF_STEPS, MF_L2, AUX_WEIGHT, INIT_STD = defaults
            init = None
            for part in kind.split("+"):                  # e.g. steps:7000+l2:0.08
                name, _, val = part.partition(":")
                if name == "steps":
                    MF_STEPS = int(val)
                elif name == "l2":
                    MF_L2 = float(val)
                elif name == "aux":
                    AUX_WEIGHT = float(val)
                elif name == "std":
                    INIT_STD = float(val)
                elif name == "ppmi":
                    init = ppmi_svd_vectors(fit, day, n_items, MF_DIM, float(val) if val else 0.05)
                elif name != "base":
                    raise ValueError(f"unknown tuning setting {part!r}")
            s = averaged(fit_l, train_ev, clk, n_users, n_items, tune_seeds, init,
                         lambda m: score(m, fit_l, target))
            log(f"=== {kind:28s} {tune_seeds} seed(s): days 0-{day-1} -> day {day} MF AUC "
                f"{roc_auc(s[:, 0], fit['label'][rows]):.5f}")
        MF_STEPS, MF_L2, AUX_WEIGHT, INIT_STD = defaults
        return

    log(f"  MF: history up to {HIST_LONG} clicks, {N_SEEDS} seeds, pretraining {pretrain}")
    mf_fit = np.full((len(ev_day) * EV_SIZE, len(MF_FEATURES)), np.nan, dtype=np.float32)
    mf_test = None
    for d in range(FIRST_MODEL_DAY, cutoff + 1):
        train_ev = np.flatnonzero(ev_day < d)
        clk = other_clicks_examples(fit, d, HIST_LONG)
        init = init_for(pretrain, fit, d, n_items)
        if d < cutoff:
            target = np.flatnonzero(ev_day == d)
            log(f"  MF trained on days 0-{d-1} -> scoring fit day {d}")
            rows = (target[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
            mf_fit[rows] = averaged(fit_l, train_ev, clk, n_users, n_items, N_SEEDS, init,
                                    lambda m: score(m, fit_l, target))
            log(f"    AUC on day {d}: {roc_auc(mf_fit[rows, 0], fit['label'][rows]):.5f}")
        else:
            log(f"  MF trained on days 0-{d-1} -> scoring the test events")
            mf_test = averaged(fit_l, train_ev, clk, n_users, n_items, N_SEEDS, init,
                               lambda m: score(m, test_l, np.arange(len(test["user"]))))
    np.savez(out_file, mf_fit=mf_fit, mf_test=mf_test)


# =================================================================== LightGBM (own process)

def _grouped(x):
    return x.reshape(-1, EV_SIZE)


def _flat(cols):
    return np.stack([np.asarray(x).reshape(-1) for x in cols], 1).astype(np.float32)


def relative_base(base):
    s = _grouped(base[:, BASE_FEATURES.index("sim_sum")])
    c = _grouped(base[:, BASE_FEATURES.index("item_ctr_logit")])
    return _flat([s - s.max(1, keepdims=True), rankdata(-s, axis=1), c - c.max(1, keepdims=True)])


def relative_mf(mf):
    m = _grouped(mf[:, 0])
    m_max = m.max(1, keepdims=True)
    ex = np.exp(m - m_max)
    lse = m_max + np.log(ex.sum(1, keepdims=True))
    return _flat([m - m_max, rankdata(-m, axis=1), ex / ex.sum(1, keepdims=True),
                  np.broadcast_to(lse, m.shape)])


def feature_matrix(d, mf, rows, use_mf=True):
    """Feature matrix for the given rows (whole events only); returns (rows, X, names)."""
    ev_idx = np.unique(rows // EV_SIZE)
    full = (ev_idx[:, None] * EV_SIZE + np.arange(EV_SIZE)).ravel()
    if not np.array_equal(np.sort(rows), full):
        raise ValueError("feature_matrix needs complete events")
    base = np.concatenate([d["cand"][full], np.repeat(d["ev"][ev_idx], EV_SIZE, axis=0)], 1)
    if not use_mf:
        return full, np.concatenate([base, relative_base(base)], 1), BASE_FEATURES + REL_FEATURES[4:]
    m = mf[full]
    if np.isnan(m).any():
        raise ValueError("missing MF scores for some rows")
    X = np.concatenate([base, m, relative_mf(m), relative_base(base)], 1)
    return full, X, BASE_FEATURES + MF_FEATURES + REL_FEATURES


def lgb_fit(X, y, names, rounds, valid=None):
    import lightgbm as lgb
    ds = lgb.Dataset(X, y, feature_name=names, free_raw_data=False)
    kw = {}
    if valid is not None:
        kw = dict(valid_sets=[lgb.Dataset(valid[0], valid[1], reference=ds)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(LGB_PARAMS, ds, num_boost_round=rounds, **kw)


def lgbm_procedure(fit, mf_fit, cutoff, use_mf=True):
    """Trees picked by early stopping on day C-1 (fit days 2..C-2), refit on days 2..C-1."""
    row_day = np.repeat(fit["day"], EV_SIZE)
    last = cutoff - 1

    def mat(mask):
        rows, X, names = feature_matrix(fit, mf_fit, np.flatnonzero(mask), use_mf)
        return X, fit["label"][rows], names

    Xa, ya, names = mat((row_day >= FIRST_MODEL_DAY) & (row_day < last))
    Xb, yb, _ = mat(row_day == last)
    m = lgb_fit(Xa, ya, names, 3000, valid=(Xb, yb))
    n_es_days, n_all_days = last - FIRST_MODEL_DAY, last - FIRST_MODEL_DAY + 1
    n_trees = max(1, int(round(m.best_iteration * n_all_days / n_es_days)))
    log(f"  trees: {m.best_iteration} by early stopping on day {last} -> {n_trees} for days "
        f"{FIRST_MODEL_DAY}-{last} (best day-{last} AUC {m.best_score['valid_0']['auc']:.5f})")
    return lgb_fit(np.concatenate([Xa, Xb]), np.concatenate([ya, yb]), names, n_trees)


def segments(fit, test):
    fit_users = np.unique(fit["user"])
    ev_new = ~np.isin(test["user"], fit_users)
    rep = test["cand"][:, CAND_FEATURES.index("rep_any")] > 0
    off = np.repeat(test["day"], EV_SIZE)
    segs = {
        "all": np.ones(len(rep), dtype=bool),
        "user has fit events": np.repeat(~ev_new, EV_SIZE),
        "new user (no fit events)": np.repeat(ev_new, EV_SIZE),
        "empty history": np.repeat(test["hlen"] == 0, EV_SIZE),
        "candidate in history": rep,
        "candidate not in history": ~rep,
    }
    for d in np.unique(off):
        segs[f"test day {int(d)}"] = off == d
    return segs


# =================================================================== main

def main():
    global INIT_STD_DEFAULT, INIT_STD
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data" / "train.csv"))
    ap.add_argument("--evaluation", default=str(ROOT / "data" / "evaluation.csv"))
    ap.add_argument("--cutoff", type=int, default=OFFICIAL_CUTOFF,
                    help="cutoff day: 4-6 = validation inside train.csv, 7 = submission")
    ap.add_argument("--out", help="submission path (cutoff 7)")
    ap.add_argument("--pretrain", default=PRETRAIN, help="item-vector pretraining: none | ppmi[:scale]")
    ap.add_argument("--init-std", type=float, default=INIT_STD_DEFAULT,
                    help="std of the random initial item vectors (0.05 reproduces experiment 11)")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--tune", help="comma-separated pretraining kinds to compare on fit day 4 (cutoff 5)")
    ap.add_argument("--tune-seeds", type=int, default=1)
    ap.add_argument("--verify-base", help="compare the cached base features with this base cache (.npz)")
    ap.add_argument("--use-mf", help="use precomputed MF scores (.npz with mf_fit, mf_test)")
    ap.add_argument("--stage", help=argparse.SUPPRESS)       # internal: run the MF stage
    ap.add_argument("--base", help=argparse.SUPPRESS)
    ap.add_argument("--mf-out", help=argparse.SUPPRESS)
    a = ap.parse_args()
    INIT_STD_DEFAULT = INIT_STD = a.init_std

    if a.stage == "mf":
        mf_stage(a.base, a.mf_out, a.cutoff, a.pretrain)
        return
    if a.stage == "tune":
        mf_stage(a.base, None, a.cutoff, None, tune=a.tune.split(","), tune_seeds=a.tune_seeds)
        return
    if not 4 <= a.cutoff <= OFFICIAL_CUTOFF:
        ap.error("cutoff must be between 4 and 7")

    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    inputs = f"{a.train}|{Path(a.train).stat().st_size}|{a.evaluation}|{Path(a.evaluation).stat().st_size}"
    base_src = "".join(inspect.getsource(f) for f in (State, Split, build, parse_event, iter_train_events))
    base_key = code_hash(base_src, inputs, str(a.cutoff), str(HIST_MAX), str(EV_SIZE))
    base_file = cache / f"15_base_c{a.cutoff}_{base_key}.npz"
    if not base_file.exists():
        fit, test, eval_order, n_users, n_items = build(a.train, a.evaluation, a.cutoff)
        tmp = base_file.with_name(base_file.stem + ".partial.npz")
        np.savez(tmp, eval_order=eval_order, n_users=n_users, n_items=n_items,
                 **{f"tr_{k}": v for k, v in fit.items()}, **{f"ev_{k}": v for k, v in test.items()})
        os.replace(tmp, base_file)
    log(f"base features: {base_file.name}")

    if a.verify_base:
        mine, other = np.load(base_file), np.load(a.verify_base)
        bad = []
        for k in other.files:
            if k not in mine.files:
                bad.append(f"{k}: missing here")
            elif not np.array_equal(mine[k], other[k]):
                bad.append(f"{k}: differs")
        log("verify-base: " + ("IDENTICAL (" + ", ".join(sorted(other.files)) + ")" if not bad else "; ".join(bad)))
        return

    if a.tune:
        subprocess.run([sys.executable, __file__, "--stage", "tune", "--base", str(base_file),
                        "--cutoff", str(a.cutoff), "--tune", a.tune, "--tune-seeds", str(a.tune_seeds),
                        "--init-std", str(a.init_std)], check=True)
        return

    if a.use_mf:
        mf_file = Path(a.use_mf)
    else:
        mf_src = inspect.getsource(mf_stage) + inspect.getsource(other_clicks_examples) + \
            inspect.getsource(long_history) + inspect.getsource(ppmi_svd_vectors)
        settings = repr((MF_DIM, MF_L2, MF_STEPS, MF_BATCH_EVENTS, MF_LR, HIST_LONG, AUX_WEIGHT, N_NEG,
                         N_SEEDS, a.pretrain, FIRST_MODEL_DAY, a.init_std, SEED_OFFSET))
        mf_file = cache / f"15_mf_c{a.cutoff}_{code_hash(base_key, mf_src, settings)}.npz"
        if not mf_file.exists():
            tmp = mf_file.with_name(mf_file.stem + ".partial.npz")
            subprocess.run([sys.executable, __file__, "--stage", "mf", "--base", str(base_file),
                            "--mf-out", str(tmp), "--cutoff", str(a.cutoff), "--pretrain", a.pretrain,
                            "--init-std", str(a.init_std)], check=True)
            os.replace(tmp, mf_file)
    log(f"MF scores: {mf_file.name}")
    z = np.load(mf_file)
    mf_fit, mf_test = z["mf_fit"], z["mf_test"]

    fit, test, eval_order, _, _ = load_base(base_file)
    test_rows = np.arange(len(test["item"]))
    if a.cutoff < OFFICIAL_CUTOFF:
        y = test["label"]
        log(f"\nVALIDATION, cutoff day {a.cutoff}: {len(test['user']):,} test events")
        model = lgbm_procedure(fit, mf_fit, a.cutoff)
        _, X, _ = feature_matrix(test, mf_test, test_rows)
        results = {"MF score alone": mf_test[:, 0], "LightGBM + MF": model.predict(X)}
        log(f"  {'segment':28s} {'rows':>7s} " + " ".join(f"{k:>16s}" for k in results))
        for seg, mask in segments(fit, test).items():
            if mask.sum() and 0 < y[mask].mean() < 1:
                log(f"  {seg:28s} {mask.mean():7.1%} " +
                    " ".join(f"{roc_auc(p[mask], y[mask]):16.5f}" for p in results.values()))
        return

    model = lgbm_procedure(fit, mf_fit, a.cutoff)
    if not a.out:
        log("no --out given: submission not written")
        return
    _, X, _ = feature_matrix(test, mf_test, test_rows)
    pred = model.predict(X)
    if not np.all(np.isfinite(pred)) or pred.min() < 0 or pred.max() > 1:
        raise ValueError("predictions outside [0, 1]")
    by_rowid = dict(zip(test["rowid"].tolist(), pred.tolist()))
    if len(by_rowid) != len(eval_order) or set(by_rowid) != set(eval_order.tolist()):
        raise ValueError("predicted rowids do not match evaluation.csv")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rowid", "predicted_probability"])
        for rid in eval_order.tolist():
            w.writerow([rid, f"{by_rowid[rid]:.6g}"])
    log(f"done. wrote {a.out}")


if __name__ == "__main__":
    main()
