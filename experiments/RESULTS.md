# Results

Public leaderboard = ROC AUC on the public part of `evaluation.csv`. Validation = local AUC
with the procedure described in the README ("How we validate"): cutoff day 5, first event per
user at/after the cutoff, nothing tuned on the test events.

| # | Experiment | Validation AUC | Public AUC | Submission file |
|---|---|---|---|---|
| 00 | Starter (provided) | 0.618 * | 0.61996 | `00_starter.csv` |
| 01 | History features + two-stage model (event click x softmax over candidates) | 0.639 * | not uploaded † | `01_history_recency.csv` |
| 02 | MF + LightGBM | 0.741 * | 0.72344 | `02_mf_lgbm.csv` |
| 03 | 02 with the rules-compliant time split | 0.680 | 0.72793 | `03_timesplit.csv` |
| 04 | 03 with a history-only MF (no per-user embedding, l2 0.05) | 0.733 | 0.76687 | `04_history_mf.csv` |
| 05 | 04 + past clicks (incl. warm-up period) as extra MF examples, weight 0.1 | 0.760 | 0.77709 | `05_click_aug.csv` |
| 06 | 05 with the MF averaged over 5 seeds | 0.786 | 0.79599 | `06_mf_seeds.csv` |
| 07 | 06 with 10 seeds instead of 5 | not run ‡ | **0.79803** | `07_mf_seeds10.csv` |

† The 01 upload (0.60732) was an earlier version of 01 (plain logistic regression, validation
0.618 *); the file was replaced by the two-stage version afterwards, which was never uploaded.

‡ Validation run skipped to save time; fit-period check: MF AUC 0.764 (5 seeds) → 0.768 (10).

\* Older validation (fit on days 1-5, test on all day-6 events, label stats updated during the
test day, some settings chosen on day 6). Optimistic and **not comparable** with 03 onwards.

Gains so far: 03 → 04 was +0.053 validation and +0.039 public (+3.9 AUC points, not 0.39%);
04 → 05 was +0.027 validation and +0.010 public.
05 → 06 was +0.026 validation and +0.019 public.
06 → 07 was +0.002 public (more seeds: gains have flattened).

## Upload log
Limit: 10 uploads per day per team. Add a line for every upload.

| Date (IST) | File | Public AUC |
|---|---|---|
| 2026-10-21 15:34 | 00_starter.csv | 0.619958 |
| 2026-10-21 15:50 | 01_history_recency.csv (earlier version, see †) | 0.607320 |
| 2026-10-21 16:40 | 02_mf_lgbm.csv | 0.723440 |
| 2026-10-21 17:43 | 03_timesplit.csv | 0.727931 |
| 2026-10-21 19:35 | 04_history_mf.csv | 0.766866 |
| 2026-10-21 20:04 | 05_click_aug.csv | 0.777089 |
| 2026-10-21 21:05 | 06_mf_seeds.csv | 0.795989 |
| 2026-10-21 ~22:30 | 07_mf_seeds10.csv | 0.798027 |

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
- **Single MF runs are noisy.** Averaging the MF over seeds was the biggest cheap win
  (fit-period MF AUC: 1 seed 0.736, 3 seeds 0.758, 5 seeds 0.764). Stronger l2 helped a
  single run (+0.004 at 0.08) but not the average, so l2 stays at 0.05.
- **The weak early-day MFs do not hurt LightGBM.** Weighting recent days, training only on the
  latest day, an "MF source day" feature or per-day percentiles all scored within 0.0002 of
  the plain setup (fit days 2-3 → day 4: 0.7529-0.7531).
- **Tried on top of 06 and ruled out** (fit days 2-3 → day 4, LightGBM baseline 0.7863; all
  changes within noise unless stated):
  - LightGBM settings: 31/127 leaves, learning rate 0.02, feature fraction 0.5, 5 seeds
    averaged: 0.7861-0.7865.
  - Ranking objectives grouped by event (lambdarank 0.7808, rank_xendcg 0.7802): worse, they
    only learn the order inside an event but the leaderboard ranks all rows together.
  - Older MF variants as extra features (03 per-user MF, 04 MF without click log): 0.7864-0.7866.
  - Candidate position in the event: click rate, item popularity and MF score are flat across
    the 10 positions, so the order of candidates carries no signal.
  - Recency inside the MF: a learned per-position weighting overfits (0.617), a "best single
    history match" term overfits (0.565), and a single learned decay converges to no decay at
    all (0.7356 = unchanged). The simulator treats a history as an unordered set of items.
- Validation must mirror the official setup: 02 looked like 0.741 locally but its procedure
  leaked; the compliant version (03) scored 0.680 and tracked the leaderboard.

## Ideas not tried yet
- Empty-history events (~1% of rows) score ~0.55: a popularity-only fallback might help.
