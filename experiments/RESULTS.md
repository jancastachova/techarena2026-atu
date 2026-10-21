# Results

Public leaderboard = ROC AUC on the public part of `evaluation.csv`. Validation = local AUC
with the procedure described in the README ("How we validate"): cutoff day 5, first event per
user at/after the cutoff, nothing tuned on the test events.

| # | Experiment | Validation AUC | Public AUC | Submission file |
|---|---|---|---|---|
| 00 | Starter (provided) | 0.618 * | — | `00_starter.csv` |
| 01 | History features + two-stage model (event click x softmax over candidates) | 0.639 * | — | `01_history_recency.csv` |
| 02 | MF + LightGBM | 0.741 * | ~0.724 | `02_mf_lgbm.csv` |
| 03 | 02 with the rules-compliant time split | 0.680 | 0.72793 | `03_timesplit.csv` |
| 04 | 03 with a history-only MF (no per-user embedding, l2 0.05) | 0.733 | 0.76687 | `04_history_mf.csv` |
| 05 | 04 + past clicks (incl. warm-up period) as extra MF examples, weight 0.1 | 0.760 | **0.77709** | `05_click_aug.csv` |

\* Older validation (fit on days 1-5, test on all day-6 events, label stats updated during the
test day, some settings chosen on day 6). Optimistic and **not comparable** with 03 onwards.

Gains so far: 03 → 04 was +0.053 validation and +0.039 public (+3.9 AUC points, not 0.39%);
04 → 05 was +0.027 validation and +0.010 public.

## What we learned
- **Which items a user clicked matters far more than when.** Weighting history clicks by
  recency, time of day and weekday/weekend added nothing in 01.
- **At most one click per event**: modelling "does the event get a click" and "which candidate"
  separately helped the linear model (01: 0.618 → 0.639).
- **Per-user embeddings memorise.** The MF without them, representing a user only by their
  clicked items, was much better (fit-period check: 0.568 → 0.660) and also covers new users.
- **The MF is data-limited.** Each extra day of training data adds ~0.03-0.05 MF AUC, so using
  the ~0.9M logged clicks as extra examples helped (05: fit-period MF AUC 0.681 → 0.736).
- **Candidates are popularity-biased** (impressions per item: sd 102 vs ~12 if random), so
  sampled negatives must follow the impression distribution.
- Validation must mirror the official setup: 02 looked like 0.741 locally but its procedure
  leaked; the compliant version (03) scored 0.680 and tracked the leaderboard.

## Ideas not tried yet
- Fix the weak early-day MFs that LightGBM learns from (day 2 MF AUC ~0.57 vs ~0.74 for day 6).
- Recency-weighted pooling of the history inside the MF (learned, not fixed).
- LightGBM ranking objective grouped by event, parameter sweep, averaging several seeds.
- Empty-history events (~1% of rows) score ~0.55: a popularity-only fallback might help.
