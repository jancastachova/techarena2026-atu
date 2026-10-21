#!/usr/bin/env python3
"""
TechArena CTR -- experiment 01: history + recency features, two-stage model.

Builds on the starter by using the time information in `history_item_ids` /
`history_timestamps`, and by learning the weights instead of hand-picking them.

Standard library only, like the starter.

How it works:
  1. Stream train.csv in time order (the file is sorted by timestamp), then the
     evaluation events sorted by timestamp. For every event, features are
     computed ONLY from what was known before that event ("as-of" features),
     then the stats are updated. This keeps features leak-free and consistent
     between training rows and evaluation rows.
     - Clicks from every history are merged into one per-user click log. It
       feeds item popularity, an item-item co-click table and a
       "clicked A, then B" transition table.
     - Labels (`clicked`) update item CTR and user click rate, but only after
       the event has been scored.
  2. An event has at most one click, so the model has two parts:
         P(click on c) = P(event gets a click) * P(c is the one | a click happens)
     - Candidate model: a softmax over the event's 10 candidates (conditional
       logit), trained on events that had a click. Uses repeat clicks of the
       candidate and how recent they were, co-click similarity with the
       history (plain and recency-weighted), transitions, item CTR/popularity.
     - Event model: logistic regression for "this event gets a click". Uses
       user activity and click rate, time of day, and how well the candidate
       set matches the history (incl. the softmax's log-sum-exp).
  3. Validation: fit on days 1..(valid_day-1), report ROC AUC on valid_day
     (day 0 is skipped for fitting because its stats are still cold).
  4. Final model: fit on days 1..6, score evaluation.csv.

Usage:
    python3 experiments/01_history_recency/history_recency.py \
    --train data/train.csv \
    --evaluation data/evaluation.csv \
    --out submissions/01_history_recency.csv
"""
import argparse
import csv
import json
import math
import random
import sys
import time
from array import array
from collections import defaultdict

TRAIN_START = 1737331200      # 2025-01-20 00:00 UTC
DAY = 86400
HOUR = 3600.0

CTR_PRIOR = 0.063             # approx. training-wide click rate per row, used before stats exist
EVENT_CLICK_PRIOR = 0.63      # approx. share of events that contain a click
SMOOTHING_K = 50              # pseudo-count for item CTR shrinkage
USER_SMOOTHING_K = 5          # pseudo-count (in events) for user click-rate shrinkage
CO_WINDOW = 20                # a new click is paired with the user's previous CO_WINDOW clicks
TAU_HIST = 24 * HOUR          # decay of history clicks when weighting similarity
TAU_POP = 24 * HOUR           # decay of item popularity ("trending")
TAU_REPEAT = 24 * HOUR        # decay of a repeat click on the candidate
SIM_SCALE = 100.0             # similarities enter as log1p(SIM_SCALE * s) (they are heavy-tailed)

CAND_FEATURES = [
    # candidate already in the user's history
    "rep_any", "rep_count", "rep_recency",
    # co-click similarity with the history (log-scaled)
    "sim_sum", "sim_decay", "sim_last", "sim_max",
    # "clicked A, next clicked B" transitions from the history (log-scaled)
    "trans_decay", "trans_last",
    # item stats
    "item_ctr_logit", "item_log_impr", "item_log_pop_decay", "item_log_clicks",
]
EVENT_FEATURES = [
    # user / history
    "hist_len", "no_hist", "log_hours_since_last", "clicks_24h",
    "user_rate_logit", "user_log_events",
    # time
    "tb_1", "tb_2", "tb_3", "day_type",
    # how well the candidate set matches the history
    "max_sim_sum", "mean_sim_sum", "n_rep", "max_ctr", "mean_ctr",
    # added after the candidate model is fitted: log-sum-exp of its utilities
    "cand_lse",
]
NC = len(CAND_FEATURES)
NE = len(EVENT_FEATURES)
NE_RAW = NE - 1               # event features known at featurisation time (all but cand_lse)


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def log_sigmoid(z):
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))


def roc_auc(scores, labels):
    """ROC AUC with ties counted as one half (average ranks)."""
    order = sorted(range(len(scores)), key=scores.__getitem__)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = 0.0
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if labels[order[k]]:
                rank_sum_pos += avg_rank
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


class State:
    """Everything learned so far from the stream (only past information)."""

    def __init__(self):
        # labels-based stats
        self.item_impr = defaultdict(int)
        self.item_clk = defaultdict(int)
        self.user_events = defaultdict(int)
        self.user_events_clk = defaultdict(int)
        self.total_rows = 0
        self.total_clk = 0
        # click log built from histories
        self.user_log = {}                       # user -> list of (ts, item), ascending, recent tail kept
        self.item_n = defaultdict(int)           # clicks per item seen in the click log
        self.pop_val = {}                        # item -> decayed click count at pop_t
        self.pop_t = {}
        self.co = defaultdict(lambda: defaultdict(int))     # item -> item -> co-click count
        self.trans = defaultdict(lambda: defaultdict(int))  # a -> b: user clicked b right after a
        self.trans_out = defaultdict(int)

    def add_history(self, user, hist_items, hist_ts):
        """Merge the clicks of an event's history into the user's click log."""
        log = self.user_log.get(user)
        if log is None:
            log = []
            self.user_log[user] = log
        last_ts = log[-1][0] if log else -1
        same_ts = {it for t, it in log if t == last_ts} if log else set()
        seen_now = set()
        for it, t in zip(hist_items, hist_ts):
            if t < last_ts or (t == last_ts and it in same_ts) or (t, it) in seen_now:
                continue
            seen_now.add((t, it))
            self._add_click(log, it, t)

    def _add_click(self, log, b, tb):
        self.item_n[b] += 1
        # decayed popularity (clicks can arrive slightly out of time order across users)
        if b in self.pop_t:
            pt = self.pop_t[b]
            if tb >= pt:
                self.pop_val[b] = self.pop_val[b] * math.exp(-(tb - pt) / TAU_POP) + 1.0
                self.pop_t[b] = tb
            else:
                self.pop_val[b] += math.exp(-(pt - tb) / TAU_POP)
        else:
            self.pop_val[b] = 1.0
            self.pop_t[b] = tb
        # co-clicks with the user's recent clicks
        co_b = self.co[b]
        for _, a in log[-CO_WINDOW:]:
            if a != b:
                co_b[a] += 1
                self.co[a][b] += 1
        # sequential transition from the previous click
        if log:
            a = log[-1][1]
            self.trans[a][b] += 1
            self.trans_out[a] += 1
        log.append((tb, b))
        if len(log) > 2 * CO_WINDOW:
            del log[:-CO_WINDOW]

    def add_labels(self, user, items, labels):
        clicked = 0
        for it, y in zip(items, labels):
            self.item_impr[it] += 1
            self.item_clk[it] += y
            self.total_rows += 1
            self.total_clk += y
            clicked |= y
        self.user_events[user] += 1
        self.user_events_clk[user] += clicked

    def event_features(self, user, ts, time_bucket, day_type, hist_items, hist_ts, items):
        """Returns (candidate feature rows, event feature row) using only past info."""
        g_ctr = self.total_clk / self.total_rows if self.total_rows else CTR_PRIOR

        # --- history items: weights and lookup tables, computed once per event
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
            # repeat click on the candidate
            cnt = hist_count.get(c, 0)
            rep_rec = math.exp(-max(0.0, ts - last_seen[c]) / TAU_REPEAT) if cnt else 0.0

            # co-click similarity and transitions
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

            # item stats
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

        # --- event level
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


class Data:
    """Flat float32 storage of candidate rows and events."""

    def __init__(self):
        self.cand = array("f")        # rows x NC
        self.ev = array("f")          # events x NE_RAW
        self.ev_start = array("i")    # first row of each event
        self.ev_size = array("b")
        self.labels = array("b")      # per row (training only)
        self.ev_day = array("b")      # per event (training only)
        self.rowids = []              # per row (evaluation only)

    def add(self, cand, ev):
        self.ev_start.append(len(self.cand) // NC)
        self.ev_size.append(len(cand))
        for r in cand:
            self.cand.extend(r)
        self.ev.extend(ev)

    @property
    def n_events(self):
        return len(self.ev_start)

    def rows(self, e):
        s = self.ev_start[e]
        return range(s, s + self.ev_size[e])

    def cand_x(self, i):
        return self.cand[i * NC:(i + 1) * NC]

    def ev_x(self, e):
        return self.ev[e * NE_RAW:(e + 1) * NE_RAW]


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


def build_features(train_path, eval_path):
    st = State()
    tr = Data()
    prev_ts = -1
    t0 = time.time()

    print("pass 1/2: streaming train.csv ...", file=sys.stderr)
    for n_ev, (ix, rows) in enumerate(iter_train_events(train_path)):
        user, ts, tb, dt, h_items, h_ts, items = parse_event(rows, ix)
        if ts < prev_ts:
            raise ValueError("train.csv is not sorted by timestamp")
        prev_ts = ts
        day = (ts - TRAIN_START) // DAY
        if not 0 <= day <= 6:
            raise ValueError(f"unexpected training timestamp {ts}")
        ys = [int(r[ix["clicked"]]) for r in rows]
        if sum(ys) > 1:
            raise ValueError("event with more than one click")
        st.add_history(user, h_items, h_ts)
        tr.add(*st.event_features(user, ts, tb, dt, h_items, h_ts, items))
        tr.labels.extend(ys)
        tr.ev_day.append(day)
        st.add_labels(user, items, ys)
        if (n_ev + 1) % 100_000 == 0:
            print(f"  {n_ev+1:,} events  ({time.time()-t0:.0f}s)", file=sys.stderr)

    print("pass 2/2: featurising evaluation.csv in time order ...", file=sys.stderr)
    with open(eval_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        ix = {c: i for i, c in enumerate(next(r))}
        eval_rows = list(r)
    events = defaultdict(list)
    for i, row in enumerate(eval_rows):
        events[row[ix["event_id"]]].append(i)
    order = sorted(events.values(), key=lambda idx: int(eval_rows[idx[0]][ix["timestamp"]]))
    ev_data = Data()
    for idx in order:
        rows = [eval_rows[i] for i in idx]
        user, ts, tb, dt, h_items, h_ts, items = parse_event(rows, ix)
        st.add_history(user, h_items, h_ts)
        ev_data.add(*st.event_features(user, ts, tb, dt, h_items, h_ts, items))
        ev_data.rowids.extend(r[ix["rowid"]] for r in rows)
    eval_order = [row[ix["rowid"]] for row in eval_rows]
    print(f"features done ({time.time()-t0:.0f}s): {len(tr.labels):,} train rows, "
          f"{len(ev_data.rowids):,} evaluation rows", file=sys.stderr)
    return tr, ev_data, eval_order


def standardiser(vectors, n):
    mean = [0.0] * n
    sq = [0.0] * n
    cnt = 0
    for x in vectors:
        cnt += 1
        for j in range(n):
            mean[j] += x[j]
            sq[j] += x[j] * x[j]
    mean = [m / cnt for m in mean]
    std = [math.sqrt(max(s / cnt - m * m, 0.0)) or 1.0 for s, m in zip(sq, mean)]
    return mean, std


def fit_candidate_model(d, events, epochs=4, seed=0):
    """Conditional logit (softmax over an event's candidates), trained on events with a click."""
    clicked = [e for e in events if any(d.labels[i] for i in d.rows(e))]
    mean, std = standardiser((d.cand_x(i) for e in clicked for i in d.rows(e)), NC)
    w = [0.0] * NC
    g2 = [1e-8] * NC                  # Adagrad accumulators
    lr = 0.05
    rng = random.Random(seed)
    for ep in range(epochs):
        rng.shuffle(clicked)
        nll = 0.0
        for e in clicked:
            rows = d.rows(e)
            xs = [[(v - m) / s for v, m, s in zip(d.cand_x(i), mean, std)] for i in rows]
            us = [sum(wj * xj for wj, xj in zip(w, x)) for x in xs]
            mx = max(us)
            ex = [math.exp(u - mx) for u in us]
            tot = sum(ex)
            k = next(t for t, i in enumerate(rows) if d.labels[i])
            nll -= us[k] - mx - math.log(tot)
            grad = [0.0] * NC
            for p_e, x in zip(ex, xs):
                p = p_e / tot
                for j in range(NC):
                    grad[j] += p * x[j]
            xk = xs[k]
            for j in range(NC):
                g = grad[j] - xk[j]
                g2[j] += g * g
                w[j] -= lr * g / math.sqrt(g2[j])
        print(f"  candidate model epoch {ep+1}/{epochs}: {len(clicked):,} clicked events, "
              f"mean NLL {nll / len(clicked):.4f} (uniform = {math.log(10):.4f})", file=sys.stderr)
    return mean, std, w


def candidate_utilities(model, d, e):
    mean, std, w = model
    return [sum(wj * (v - m) / s for wj, v, m, s in zip(w, d.cand_x(i), mean, std)) for i in d.rows(e)]


def log_softmax(us):
    mx = max(us)
    lse = mx + math.log(sum(math.exp(u - mx) for u in us))
    return [u - lse for u in us], lse


def event_vector(d, e, lse):
    return list(d.ev_x(e)) + [lse]


def fit_event_model(d, events, cand_model, epochs=3, seed=0):
    """Logistic regression: does this event get a click?"""
    vecs = []
    ys = []
    for e in events:
        _, lse = log_softmax(candidate_utilities(cand_model, d, e))
        vecs.append(event_vector(d, e, lse))
        ys.append(1 if any(d.labels[i] for i in d.rows(e)) else 0)
    mean, std = standardiser(vecs, NE)
    xs = [[(v - m) / s for v, m, s in zip(x, mean, std)] for x in vecs]
    w = [0.0] * NE
    b = logit(sum(ys) / len(ys))
    g2 = [1e-8] * NE                  # Adagrad accumulators
    g2_b = 1e-8
    lr = 0.05
    order = list(range(len(xs)))
    rng = random.Random(seed)
    for ep in range(epochs):
        rng.shuffle(order)
        loss = 0.0
        for k in order:
            x = xs[k]
            p = sigmoid(b + sum(wj * xj for wj, xj in zip(w, x)))
            y = ys[k]
            loss -= math.log(max(p if y else 1.0 - p, 1e-15))
            g = p - y
            g2_b += g * g
            b -= lr * g / math.sqrt(g2_b)
            for j in range(NE):
                gj = g * x[j]
                g2[j] += gj * gj
                w[j] -= lr * gj / math.sqrt(g2[j])
        print(f"  event model epoch {ep+1}/{epochs}: {len(xs):,} events, "
              f"mean log-loss {loss / len(xs):.5f}", file=sys.stderr)
    return mean, std, w, b


def score_events(d, events, cand_model, ev_model):
    """Yields (row index, log P(click on this candidate)) for every row of the events."""
    mean, std, w, b = ev_model
    for e in events:
        log_p, lse = log_softmax(candidate_utilities(cand_model, d, e))
        x = event_vector(d, e, lse)
        z = b + sum(wj * (v - m) / s for wj, v, m, s in zip(w, x, mean, std))
        lp_event = log_sigmoid(z)
        for i, lp in zip(d.rows(e), log_p):
            yield i, lp_event + lp


def fit(d, events):
    cand_model = fit_candidate_model(d, events)
    ev_model = fit_event_model(d, events, cand_model)
    return cand_model, ev_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--valid-day", type=int, default=6, help="training day (1-6) held out for validation")
    ap.add_argument("--skip-validation", action="store_true")
    a = ap.parse_args()

    tr, ev_data, eval_order = build_features(a.train, a.evaluation)

    if not a.skip_validation:
        vd = a.valid_day
        fit_ev = [e for e in range(tr.n_events) if 1 <= tr.ev_day[e] < vd]
        val_ev = [e for e in range(tr.n_events) if tr.ev_day[e] == vd]
        print(f"validation: fit on days 1-{vd-1} ({len(fit_ev):,} events), "
              f"test on day {vd} ({len(val_ev):,} events)", file=sys.stderr)
        cand_model, ev_model = fit(tr, fit_ev)
        idx, scores = zip(*score_events(tr, val_ev, cand_model, ev_model))
        y_val = [tr.labels[i] for i in idx]
        print(f"VALIDATION AUC (day {vd}): {roc_auc(list(scores), y_val):.5f}", file=sys.stderr)
        print("single-feature AUCs on the validation day, and model weights:", file=sys.stderr)
        for j, name in enumerate(CAND_FEATURES):
            auc = roc_auc([tr.cand[i * NC + j] for i in idx], y_val)
            print(f"  cand  {name:22s} {max(auc, 1 - auc):.4f}{'' if auc >= 0.5 else ' (inverse)'}"
                  f"   weight {cand_model[2][j]:+.3f}", file=sys.stderr)
        for j, name in enumerate(EVENT_FEATURES):
            print(f"  event {name:22s}   weight {ev_model[2][j]:+.3f}", file=sys.stderr)

    print("final model: fit on days 1-6 ...", file=sys.stderr)
    cand_model, ev_model = fit(tr, [e for e in range(tr.n_events) if tr.ev_day[e] >= 1])
    preds = {}
    for i, lp in score_events(ev_data, range(ev_data.n_events), cand_model, ev_model):
        preds[ev_data.rowids[i]] = math.exp(lp)
    if len(preds) != len(eval_order):
        raise ValueError("prediction count does not match evaluation rows")
    with open(a.out, "w", newline="", encoding="utf-8") as out_f:
        w = csv.writer(out_f)
        w.writerow(["rowid", "predicted_probability"])
        for rid in eval_order:
            w.writerow([rid, f"{preds[rid]:.6g}"])
    print(f"done. wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
