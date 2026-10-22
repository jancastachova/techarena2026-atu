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
| 07 | 06 with 10 seeds instead of 5 | not run ‡ | 0.79803 | `07_mf_seeds10.csv` |
| 08 | 07 with a choice-model loss in the MF (one pick or no click per event, scale 0.1) | 0.790 | 0.79750 | `08_choice_mf.csv` |
| 09 | 07 with the user vector over ALL known clicks before the event (up to 100), not the last 20 | not run ‡ | 0.80493 | `09_full_history.csv` |
| 10 | 09 with the user's OTHER clicks (not only earlier ones) as click-log context | not run ‡ | 0.81297 | `10_other_clicks.csv` |
| 11 | 10 with one shared vector per item (history and candidate roles) | not run ‡ | **0.81584** | `11_tied_items.csv` |

† The 01 upload (0.60732) was an earlier version of 01 (plain logistic regression, validation
0.618 *); the file was replaced by the two-stage version afterwards, which was never uploaded.

‡ Validation run skipped to save time. Fit-period checks instead: 07 MF AUC 0.764 (5 seeds)
→ 0.768 (10 seeds); 09 day-6 LightGBM AUC 0.8099 (07) → 0.8236; 10: → 0.8307; 11: → 0.8336.

\* Older validation (fit on days 1-5, test on all day-6 events, label stats updated during the
test day, some settings chosen on day 6). Optimistic and **not comparable** with 03 onwards.

Gains so far: 03 → 04 was +0.053 validation and +0.039 public (+3.9 AUC points, not 0.39%);
04 → 05 was +0.027 validation and +0.010 public.
05 → 06 was +0.026 validation and +0.019 public.
06 → 07 was +0.002 public (more seeds: gains have flattened).
07 → 08 was −0.0005 public (within noise): the choice loss did not carry over.
07 → 09 was +0.007 public (day-6 LightGBM, fit period: 0.8099 → 0.8236).
09 → 10 was +0.008 public (day-6 LightGBM, fit period: 0.8236 → 0.8307).
10 → 11 was +0.003 public (day-6 LightGBM, fit period: 0.8307 → 0.8336).

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
| 2026-10-22 ~00:00 | 08_choice_mf.csv | 0.797495 |
| 2026-10-22 ~08:40 | 09_full_history.csv | 0.804933 |
| 2026-10-22 ~10:25 | 10_other_clicks.csv | 0.812971 |
| 2026-10-22 ~12:40 | 11_tied_items.csv | 0.815842 |

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
- **A choice-model loss improves the MF but not the final model.** Treating each event as
  "pick one of 10 or nothing" made the MF alone clearly better (fit-period MF AUC, 3 seeds:
  0.758 → 0.765; cutoff-5 MF alone 0.765 → 0.776), but after LightGBM it was a tie (day-6
  0.8093 vs 0.8099 for 07) and public dropped slightly (0.79803 → 0.79750). LightGBM had
  already rebuilt the one-pick-per-event structure from the event-relative MF features.
- **Also ruled out on top of 06/07:** strongly regularised per-user embeddings on top of the
  history (fit-period MF AUC 0.731-0.735 vs 0.7356), and a "shown before, not clicked" signal
  (those candidates are clicked at 6.9%, same as any other; only "shown before and clicked",
  i.e. repeat clicks, matters, which the history already covers).
- **Use every known click, not just the 20 in the history.** ~60% of rows have a full
  20-click history, so older clicks were cut off; since order and timing do not matter, they
  are as informative as recent ones. Representing the user by all logged clicks before the
  event (up to 100; mean 26 instead of 17) gave fit-period MF AUC 0.736 → 0.752 (1 seed;
  50 clicks 0.752, so it levels off), day-6 LightGBM 0.8099 → 0.8236, public +0.007.
  Check: the 20 most recent logged clicks equal the event's own history for 99.94% of events.
- **Learn item vectors from each click's OTHER clicks, not only the earlier ones.** Since a
  history is a set, every other click of the user (before the time limit) is valid context for
  a click-log example: mean context 12.6 → 23.5 clicks, fit-period MF AUC 0.752 → 0.779
  (1 seed), 0.772 → 0.791 (3 seeds), day-6 LightGBM 0.8236 → 0.8307, public +0.008.
  Re-tuning on top of 09 found nothing else: dim 64, 8000 steps, l2 0.03/0.08, click-log
  weight 0.05/0.2/0.3/0.5 all equal or worse (l2 0.08 looked +0.003 with 1 seed, −0.001 with 3).
  Repeat candidates that are only in clicks older than the 20-click window are real (clicked
  28%) but rare (0.05% of rows).
- **Rare items are the weak spot.** Fit day 4 (10's MF, 1 seed) by the candidate's clicks so
  far: 1-4 clicks AUC 0.68 (4% of rows), 5-9 0.73 (16%), 10-19 0.77 (33%), 20-39 0.80 (28%),
  40+ 0.81 (20%); the same pattern for users with few clicks. "Does this event get any click"
  is hard (AUC 0.63 from the best candidate's score).
- **One vector per item** (instead of separate history and candidate vectors) gives each item
  twice the training signal: fit-period MF AUC 0.779 → 0.784 (1 seed), 0.791 → 0.795 (3 seeds),
  largest gain on the earliest days (least data); day-6 LightGBM 0.8307 → 0.8336, public +0.003.
- **Tried on top of 10, not kept:** 49 instead of 9 click-log negatives (0.774 vs 0.779), and
  the user's other clicks (own click removed) as the training user vector for impressions
  (0.7793 vs 0.7790).
- **Tried on top of 11, not kept** (fit period; MF tests on day 4 with 1 seed unless stated,
  baseline 0.7838; LightGBM tests on day 6, baseline 0.8336):
  - Re-tuning: dim 64 0.772, l2 0.03 0.779, l2 0.08 0.788 with 1 seed but 0.7936 vs 0.7946 with
    3 seeds (third time a single-seed l2 gain vanished), l2 0.12 0.784, click-log weight 0.05/0.2
    0.785/0.780, 8000 steps 0.773.
  - Candidate set as a hint about the user: candidates ARE mildly personalised (cosine to the
    user's history 0.034 vs −0.001 for random popular items), but adding the other 9 candidates
    to the user vector gives 0.756 (own vectors, overfits) or 0.7843 (shared vectors, noise).
  - Candidate-aware attention over the history (starts as plain averaging): 0.7837, although it
    learned some focus. A user's taste behaves like a single direction, not several interests.
  - Item drift: item click rates agree slightly less across time halves (0.158) than across
    random halves (0.184): slow drift, too small to matter.
  - LightGBM with 10's and 11's MF features together: 0.8339; with full-history click count and
    repeat flags: 0.8337. Both noise.
- Validation must mirror the official setup: 02 looked like 0.741 locally but its procedure
  leaked; the compliant version (03) scored 0.680 and tracked the leaderboard.

## Ideas not tried yet
- Empty-history events (~1% of rows) score ~0.55: a popularity-only fallback might help.
