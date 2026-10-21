#!/usr/bin/env python3
"""
TechArena CTR -- simple starter solution given from the challenge link.
(https://portal-techarena-ireland.hackathon.com/submit.html)

Uses only the Python standard library (csv, json, math, collections) -- no numpy,
pandas, or scikit-learn required. Intentionally simple: it should get you onto the
leaderboard with a clearly-beatable score, not win the competition.

Idea (three ingredients, combined by hand):
  1. Item popularity:  smoothed empirical click-through rate of each item.
  2. User propensity:  smoothed empirical click-through rate of each user.
  3. "People who clicked A also clicked B": a simple item-item co-click similarity,
     built from what users clicked/saw in their history, used to score each candidate
     against the user's own click history (this is a basic collaborative-filtering
     signal -- it is what gives this script most of its lift over popularity alone).

(1) and (2) are combined in log-odds ("naive Bayes") space, and (3) is added as a
bonus term. None of this uses the structure of the actual data-generating model --
a good participant solution should do meaningfully better by modeling user/item
interactions directly (e.g. matrix factorization) instead of hand-combining
independent signals like this.

Usage:
    python3 starter_solution.py --train train.csv --evaluation evaluation.csv --out submission.csv
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict

SMOOTHING_K = 50          # pseudo-count for item/user CTR shrinkage toward the global rate
CF_WEIGHT = 25.0          # how much the co-click similarity bonus counts vs. the CTR priors


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def read_rows(path, has_label):
    """Yields dicts for every row of a train/evaluation CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["history_item_ids"] = json.loads(row["history_item_ids"])
            if has_label:
                row["clicked"] = int(row["clicked"])
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    # ---- pass 1 over train.csv: item/user click + impression counts, and an
    #      implicit "what did this user interact with" set (clicks + history) for
    #      the co-click similarity table -------------------------------------
    item_clicks, item_impr = defaultdict(int), defaultdict(int)
    user_clicks, user_impr = defaultdict(int), defaultdict(int)
    user_interacted = defaultdict(set)
    total_clicks, total_rows = 0, 0

    print("pass 1/2: reading train.csv ...", file=sys.stderr)
    for i, row in enumerate(read_rows(a.train, has_label=True)):
        uid, iid, y = row["user_id"], row["item_id"], row["clicked"]
        item_clicks[iid] += y
        item_impr[iid] += 1
        user_clicks[uid] += y
        user_impr[uid] += 1
        total_clicks += y
        total_rows += 1
        if row["history_item_ids"]:
            user_interacted[uid].update(row["history_item_ids"])
        if y:
            user_interacted[uid].add(iid)
        if (i + 1) % 2_000_000 == 0:
            print(f"  {i+1:,} rows", file=sys.stderr)

    global_ctr = total_clicks / total_rows
    item_ctr = {iid: (item_clicks[iid] + SMOOTHING_K * global_ctr) / (item_impr[iid] + SMOOTHING_K)
                for iid in item_impr}
    user_ctr = {uid: (user_clicks[uid] + SMOOTHING_K * global_ctr) / (user_impr[uid] + SMOOTHING_K)
                for uid in user_impr}
    g_logit = logit(global_ctr)

    # ---- item-item co-click table: co[a][b] = how many users interacted with both
    #      a and b. This is the "people who liked A also liked B" signal. --------
    print("building item-item co-click table ...", file=sys.stderr)
    item_pop = defaultdict(int)           # how many users interacted with each item
    co = defaultdict(lambda: defaultdict(int))
    for uid, items in user_interacted.items():
        items = list(items)
        for it in items:
            item_pop[it] += 1
        for x in range(len(items)):
            for yix in range(x + 1, len(items)):
                a_, b_ = items[x], items[yix]
                co[a_][b_] += 1
                co[b_][a_] += 1
    item_pop_sqrt = {k: math.sqrt(v) for k, v in item_pop.items()}

    def cf_score(candidate_item, history_items):
        """Cosine-normalized co-click similarity between the candidate and the
        user's own history items (0 if the candidate has no measured co-click
        overlap with anything in the history)."""
        if candidate_item not in co or not history_items:
            return 0.0
        neighbors = co[candidate_item]
        total = 0.0
        denom_a = item_pop_sqrt.get(candidate_item, 0.0)
        for h in history_items:
            c = neighbors.get(h)
            if c:
                denom_b = item_pop_sqrt.get(h, 0.0)
                if denom_a and denom_b:
                    total += c / (denom_a * denom_b)
        return total

    def score_row(uid, iid, history_items):
        i_ctr = item_ctr.get(iid, global_ctr)
        u_ctr = user_ctr.get(uid, global_ctr)
        combo_logit = logit(i_ctr) + logit(u_ctr) - g_logit
        return combo_logit + CF_WEIGHT * cf_score(iid, history_items)

    # ---- pass 2: score evaluation.csv and write the submission ------------------
    print("pass 2/2: scoring evaluation.csv ...", file=sys.stderr)
    with open(a.out, "w", newline="", encoding="utf-8") as out_f:
        w = csv.writer(out_f)
        w.writerow(["rowid", "predicted_probability"])
        for i, row in enumerate(read_rows(a.evaluation, has_label=False)):
            z = score_row(row["user_id"], row["item_id"], row["history_item_ids"])
            p = 1.0 / (1.0 + math.exp(-z))   # back to a probability for the submission format
            w.writerow([row["rowid"], f"{p:.6g}"])
            if (i + 1) % 2_000_000 == 0:
                print(f"  {i+1:,} rows scored", file=sys.stderr)

    print(f"done. wrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
