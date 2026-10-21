---
tags:
  - hackathon/huawei-tech-arena/challenge-0
---


# Predict click-through rates

You'll predict click-through probability on a synthetic ad/recommendation dataset: each event shows one user 10 candidate items, at most one of which gets clicked, and your job is to estimate the probability of a click for every candidate in the held-out evaluation set. The training data includes each user's recent click history (up to 20 prior clicks, including from a warm-up period before the training window), giving you a real personalization signal to exploit alongside standard popularity and propensity features.

Submissions are ranked by global ROC AUC. The dataset is sized to be workable on a laptop (under 15 minutes to train a solid baseline, no GPU required), so focus your time on modeling rather than infrastructure.

# Instructions
## TechArena 2026 – Phase 1: Click Prediction

## Task

Phase 1 is a click-probability prediction task on synthetic recommendation data. No real users or products are involved. Your task is to predict a click probability for every row in `evaluation.csv`.

- An **event** is one recommendation opportunity: one user is shown **10 candidate items** at one moment.
- A **row** in the CSV files is one candidate (an *impression*): one item shown to one user in one event.
- In the training data the target `clicked` is 1 if that candidate was clicked and 0 otherwise.
- An event has **at most one clicked candidate**. Events with no click are also possible: all 10 of their rows have `clicked = 0`.

For **every row of `evaluation.csv`**, predict the probability that the candidate was clicked. Do not submit an extra row or
value for the "no click" outcome.

## Files

| File                     | Content                                                                       | Data rows (header excluded) |
| ------------------------ | ----------------------------------------------------------------------------- | --------------------------- |
| `train.csv`              | Labelled candidates (includes `clicked`), 686,824 events                      | 6,868,240                   |
| `evaluation.csv`         | Candidates to predict, no labels, 52,939 events (one per user)                | 529,390                     |
| `example_submission.csv` | Format example: one line per evaluation row, all with the same constant value | 529,390                     |
| `README.txt`             | Short description shipped with the data                                       | –                           |

`train.csv` is about 7.3 GB. Consider streaming or chunked processing. Labels for the evaluation rows are not provided.

## Columns

`evaluation.csv` has the same columns as `train.csv` except `clicked`.

| Column               | Meaning                                                                                            |
| -------------------- | -------------------------------------------------------------------------------------------------- |
| `rowid`              | Identifier of the row (one candidate in one event). The key of your submission.                    |
| `event_id`           | Identifier of the event. All 10 rows of an event share user, timestamp and history.                |
| `user_id`            | The user.                                                                                          |
| `item_id`            | The candidate item shown in this row.                                                              |
| `timestamp`          | Event time, Unix seconds, UTC (whole minutes)                                                      |
| `time_bucket`        | Six-hour block of the UTC day: 0 = 00:00–05:59, 1 = 06:00–11:59, 2 = 12:00–17:59, 3 = 18:00–23:59. |
| `day_type`           | 0 = Monday–Friday, 1 = Saturday or Sunday (UTC date).                                              |
| `history_item_ids`   | JSON array of items this user clicked before the event.                                            |
| `history_timestamps` | JSON array of the Unix-second times of those clicks.                                               |
| `clicked`            | **Training only.** 1 = clicked, 0 = not clicked.                                                   |

- `rowid`, `event_id`, `user_id` and `item_id` are **opaque identifiers**. Treat them as text, not numbers, and do not try to
  interpret them. `rowid` and `event_id` are random and carry no information; use them as keys only.
- The history columns hold JSON arrays inside quoted CSV fields. Read the files with a CSV parser, then parse these fields
  with a JSON parser; do not split lines on commas.

## History and time split

- A history contains previous **clicks** only (not all earlier impressions): the latest **20 at most**, strictly before the current event, ordered **oldest to newest**.
- `history_item_ids` and `history_timestamps` are aligned element by element and have the same length. An empty history is `[]`. The same item can appear several times.
- The simulated timeline starts before the training period (a *warm-up* period, before 2025-01-20 UTC). Its impression rows are
  **not** included, but its clicks can appear in histories. A user can have a non-empty history without any row in `train.csv`.
- **Training** covers events from 2025-01-20 00:00 UTC up to, not including, 2025-01-27 00:00 UTC (the cutoff).
- **Evaluation** contains, for each user, the **first event at or after the cutoff**, with all its 10 candidates. Later events of these users are not included.
- Some evaluation users do not appear in `train.csv`, and some evaluation events have an empty history. Make sure your predictions cover these cases too.

**Local validation.** You have no evaluation labels, so validate on a later part of the training period: fit on earlier events, test on later ones, and keep all rows of an event on the same side of the split. Compute features and aggregate statistics (for example click rates per user or item) using labels from the fitting part only, never from the events you are testing on.

A later training window does not exactly reproduce the official setup, which uses one first event per user.

## Submission format

A UTF-8 CSV file with exactly this header (case-sensitive, this order):

```
rowid,predicted_probability
```

Format illustration only – a real submission has one line for **every** evaluation row:

```
rowid,predicted_probability
row_aaaa0000,0.0629
row_bbbb1111,0.1200
row_cccc2222,0.0003
```

Requirements:

- Every `rowid` of `evaluation.csv` appears **exactly once**: no missing, duplicate or unknown rowids.
- Exactly the two columns above and no extra index column. Keep each `rowid` exactly as given (no trimming or case changes).
- Each prediction is a finite number in **[0, 1]**. Plain decimals (`0.25`, `.5`, `1`) and scientific notation (`2e-3`) are accepted.
  Empty values, text, `nan` and `inf` are rejected.
- Predictions are matched to rows **by `rowid`**, so row order does not matter. Keeping the order of `example_submission.csv`
  is still convenient.
- The probabilities of an event do **not** have to sum to one.

Upload the finished file through the competition's submission page, then check its processing status in your submission
history. If an upload is interrupted, check the history before uploading again.

## Evaluation

Submissions are ranked by **ROC AUC**; higher is better. AUC measures how well your predictions rank clicked rows above
non-clicked rows: it is the probability that a randomly chosen clicked row gets a higher prediction than a randomly chosen
non-clicked row (ties count as one half). Random ranking has an expected AUC of 0.5 and perfect separation gives 1.0. No
threshold is needed, and only the ranking matters, not calibration. The example submission is constant and therefore scores ~0.5.

The evaluation rows are split between a public leaderboard and private leaderboard set randomly.
Submission scoring computes the ROC AUC for both, but only the public leaderboard score will get updated during Phase 1 of the competition.
The final ranking will be determined by the private AUC, to be revealed only after the end of Phase 1.
ROC AUC is computed over all rows in the public/private set rows, and is not a per-user or per-event average.

**Each team is limited to 10 submissions per day!**

## Starter solution

A simple starter solution is provided in starter_solution.py.

To run it yourself, use the command
```
python3 starter_solution.py --train train.csv --evaluation evaluation.csv --out submission.csv
```

## Checklist

- [ ] Read the CSV files with a CSV parser and parse the history columns as JSON.
- [ ] Keep identifiers as text and keep each prediction attached to its `rowid`.
- [ ] Produce one prediction for every evaluation row.
- [ ] Check the header, the number of rows, duplicates, missing or unknown `rowid`s, and that all values are in [0, 1].
- [ ] Check your submission status before retrying an upload.


# Example Submission
|                                          |                           |
| ---------------------------------------- | ------------------------- |
| **rowid**                                | **predicted_probability** |
| **row_fa9215e840b18b0cdca93d722e7e0340** | 0.06290156                |
| **row_4ba616ef837bd39bff2d62b8bf347441** | 0.06290156                |
| **row_cc27c47eac23dc1f98893ce0529e53aa** | 0.06290156                |
| **row_e7fac381380db823900ef78b5f5b0f83** | 0.06290156                |
| **row_5612378b37dd1708ce4a009bd9f94692** | 0.06290156                |
| **row_aef8265c60c764cad92e144121e79d39** | 0.06290156                |
| **row_87c1138c76b153885bb95316c3d22c2d** | 0.06290156                |
| **row_9c4c6eafae1f24f92c9bd748b5bdddac** | 0.06290156                |
| **row_0ac0420d006c8ee91b621dba23e781fc** | 0.06290156                |
| **row_5a1371c55bd03540674997a176ba493c** | 0.06290156                |
| **row_2f6168f8a070d65a422849c16ad74b82** | 0.06290156                |
| **row_d0ea4a85f31a4c35e4e372202d1030f9** | 0.06290156                |
| **row_342df5e6f2b480d449f025a2f10f8427** | 0.06290156                |
| **row_0009e8d8c8c6b0774fd33949fca46b85** | 0.06290156                |
| **row_c938928e9b9fd0e8f19e43ff2a6309ef** | 0.06290156                |
| **row_f7a383de35ddfebbeb689ef9527a5839** | 0.06290156                |
| **row_48b12ee8effe63c6fba57676401883fa** | 0.06290156                |
| **row_6f726c4c933ed15e71f03b4c2f4d9adf** | 0.06290156                |
| **row_c7a1417b1a9baf7ee8af527b1573dd8f** | 0.06290156                |
| **row_200f199e24fa91dd11c4c23124abc59b** | 0.06290156                |
| **row_8f223cd2b5bd7e2d6ad2544af23910f8** | 0.06290156                |


# Sample of the train set
| Field                | Example            | Meaning                                                                                                                                        |
| -------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `rowid`              | `row_5e3f…`        | Unique ID for this row. It’s the key your submission uses (`rowid,predicted_probability`).                                                     |
| `event_id`           | `evt_c892…`        | One event: one user shown 10 candidate items. Each event fans out into 10 rows, one per candidate.                                             |
| `user_id`            | `usr_a5e4…`        | Opaque user identifier.                                                                                                                        |
| `item_id`            | `itm_84af…`        | The candidate item shown in this row. This is what you’re predicting a click for.                                                              |
| `timestamp`          | `1737331200`       | Unix seconds for when the event happened. This one is 2025-01-20 00:00 UTC, a Monday.                                                          |
| `time_bucket`        | `0`                | Six-hour UTC block: `0` = 00:00–05:59.                                                                                                         |
| `day_type`           | `0`                | `0` = weekday (Mon–Fri), `1` = weekend.                                                                                                        |
| `history_item_ids`   | list of 20 `itm_…` | The user’s previous clicks (max 20), oldest first.                                                                                             |
| `history_timestamps` | list of 20 ints    | Unix times aligned one-to-one with `history_item_ids`. Here they span roughly 14–20 Jan, and the last one is about 1.5 hours before the event. |
| `clicked`            | `1`                | The label: `1` means the user clicked this candidate, `0` means they didn’t. It’s only present in train.                                       |

```json 
{ 
	"rowid": "row_5e3f316bf724083745def2ed54ffe2db", 
	"event_id": "evt_c8924233d50f30f08baaba91619fd560", 
	"user_id": "usr_a5e436d47673e294137606646dc81d15", 
	"item_id": "itm_84afdab03779ab1fc1dcb0735bd655a5", 
	"timestamp": 1737331200, 
	"time_bucket": 0, 
	"day_type": 0, 
	"clicked": 1 
} 
```

**History** (`history_item_ids` ↔ `history_timestamps`, oldest first)
For this one user:

| #   | Item                                   | Timestamp  |
| --- | -------------------------------------- | ---------- |
| 1   | `itm_de0de8f3825d609a883730a6ea449da3` | 1736868540 |
| 2   | `itm_d15678985c422ba3a67b91c4230016ac` | 1736871180 |
| 3   | `itm_4896e9cdd988c35dba002fafca8ab3dd` | 1736904900 |
| 4   | `itm_e3805eccb5ea89b9963e12343a4d54e4` | 1737011460 |
| 5   | `itm_852ab4431b36d1d5880eb06c966c26b9` | 1737028680 |
| 6   | `itm_e102cf519ee2b87849ee3c36920394fe` | 1737049200 |
| 7   | `itm_9f7e14e9f84c9309df2eab96cbf0de00` | 1737060660 |
| 8   | `itm_c13f9e1d0b43fe14c5bc44c4f352fc02` | 1737065940 |
| 9   | `itm_8019c0c6ef0ffe9e420090d72445a251` | 1737068940 |
| 10  | `itm_cb95e6b46f855a5c6c651f50de3e929f` | 1737069840 |
| 11  | `itm_5405d4a89d485c1a85ca0e61b0763c3a` | 1737105480 |
| 12  | `itm_db96034ec631648d3a55e25eec3d6bf4` | 1737106980 |
| 13  | `itm_e067b6b9d193a13fa98cefff9bf3ec56` | 1737117360 |
| 14  | `itm_3be5def487dc84103cca0a44ac017023` | 1737135180 |
| 15  | `itm_799c3a486b6fee60955cf969b6570c7a` | 1737143820 |
| 16  | `itm_1011235046664722a84248e8bd8d8b83` | 1737224640 |
| 17  | `itm_bd40f1b96f83e75a2c01af13143f4c71` | 1737268920 |
| 18  | `itm_431fac2607e0cd8987f6406ec6caa768` | 1737292500 |
| 19  | `itm_9aaa4dfb07e7fa87e05059bedf899311` | 1737306960 |
| 20  | `itm_40cf776a23627aaf353408c1bec4a43f` | 1737325980 |

```csv

rowid,event_id,user_id,item_id,timestamp,time_bucket,day_type,history_item_ids,history_timestamps,clicked
row_5e3f316bf724083745def2ed54ffe2db,evt_c8924233d50f30f08baaba91619fd560,usr_a5e436d47673e294137606646dc81d15,itm_84afdab03779ab1fc1dcb0735bd655a5,1737331200,0,0,"[""itm_de0de8f3825d609a883730a6ea449da3"",""itm_d15678985c422ba3a67b91c4230016ac"",""itm_4896e9cdd988c35dba002fafca8ab3dd"",""itm_e3805eccb5ea89b9963e12343a4d54e4"",""itm_852ab4431b36d1d5880eb06c966c26b9"",""itm_e102cf519ee2b87849ee3c36920394fe"",""itm_9f7e14e9f84c9309df2eab96cbf0de00"",""itm_c13f9e1d0b43fe14c5bc44c4f352fc02"",""itm_8019c0c6ef0ffe9e420090d72445a251"",""itm_cb95e6b46f855a5c6c651f50de3e929f"",""itm_5405d4a89d485c1a85ca0e61b0763c3a"",""itm_db96034ec631648d3a55e25eec3d6bf4"",""itm_e067b6b9d193a13fa98cefff9bf3ec56"",""itm_3be5def487dc84103cca0a44ac017023"",""itm_799c3a486b6fee60955cf969b6570c7a"",""itm_1011235046664722a84248e8bd8d8b83"",""itm_bd40f1b96f83e75a2c01af13143f4c71"",""itm_431fac2607e0cd8987f6406ec6caa768"",""itm_9aaa4dfb07e7fa87e05059bedf899311"",""itm_40cf776a23627aaf353408c1bec4a43f""]","[1736868540,1736871180,1736904900,1737011460,1737028680,1737049200,1737060660,1737065940,1737068940,1737069840,1737105480,1737106980,1737117360,1737135180,1737143820,1737224640,1737268920,1737292500,1737306960,1737325980]",1

```

# Simple Starter Solution
Uses only the Python standard library (csv, json, math, collections) -- no numpy,
pandas, or scikit-learn required. Intentionally simple: it should get you onto the
leaderboard with a clearly-beatable score, not win the competition.

Idea (three ingredients, combined by hand):

1. Item popularity: smoothed empirical click-through rate of each item.

2. User propensity: smoothed empirical click-through rate of each user.

3. "People who clicked A also clicked B": a simple item-item co-click similarity, built from what users clicked/saw in their history, used to score each candidate against the user's own click history (this is a basic collaborative-filtering signal -- it is what gives this script most of its lift over popularity alone).
 
(1) and (2) are combined in log-odds ("naive Bayes") space, and (3) is added as a bonus term. None of this uses the structure of the actual data-generating model - a good participant solution should do meaningfully better by modeling user/item interactions directly (e.g. matrix factorization) instead of hand-combining independent signals like this.

**Usage**:
```bash
python3 starter_solution.py --train train.csv --evaluation evaluation.csv --out submission.csv
```

## Source


```python
#!/usr/bin/env python3
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
```