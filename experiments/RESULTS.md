# Results

Public leaderboard = ROC AUC on the public part of `evaluation.csv`. Validation = local AUC
with the procedure described in the README ("How we validate"): cutoff day 5, first event per
user at/after the cutoff, nothing tuned on the test events.

## How we run a batch (read this file before every run)
1. **Start from the current best** (now 15 = experiment 13's pipeline, two seed sets
   averaged) and check "What we learned" below, so nothing already ruled out is retried.
2. **Test ideas on the fit period only** (official split, data before day 7, labels known):
   LightGBM trained on fit days 2-5; day 6 split in halves by event, half A for early stopping
   and blend weights, half B for the reported AUC. Baseline (13's features): **0.83571 on half
   B**. MF-only ideas: `--tune` (MF trained on days 0-5, AUC on day 6; 13 with 3 seeds 0.8175).
   Never tune on evaluation.csv or on leaderboard scores.
3. **Reusable scripts** go in `experiments/lab/` (quick checks on cached arrays) or a new
   standalone `experiments/NN_name/` folder (anything that produces a submission).
4. **Every run's output** is saved in `experiments/logs/YYYY-MM-DD_<experiment>_<what>.txt`,
   and its result is added here (table below if uploaded, "What we learned" otherwise).
5. **A submission is built only if a change beats the baseline clearly** (≥ ~+0.001 on half B;
   +0.0002 is noise level). Its file is checked with `src/check_submission.py` and handed over
   for upload; uploads are not used to test ideas.

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
| 11 | 10 with one shared vector per item (history and candidate roles) | not run ‡ | 0.81584 | `11_tied_items.csv` |
| 12 | 11 as ONE standalone file; item vectors start at std 0.005 instead of 0.05 | not run ‡ | 0.81573 | `12_standalone.csv` |
| 13 | 12 with MF training 3000 steps instead of 5000 (tuned on days 0-5 → day 6) and 15 seeds | 0.814 | 0.81784 | `13_retune_7days.csv` |
| 15 | mean of 13 (MF seeds 0-14) and a twin run with MF seeds 15-29 (`15_seeds15_29.csv`, day-6 LightGBM 0.8355) | not run ‡ | **0.81807** | `15_avg_13_15.csv` |

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
11 → 12 was −0.0001 public (within noise; day-6 LightGBM 0.83360 → 0.83359).
12 → 13 was +0.002 public (day-6 LightGBM 0.8336 → 0.8353).
13 → 15 was +0.0002 public (averaging two seed sets: the seed lever is spent).

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
| 2026-10-22 ~16:05 | 12_standalone.csv | 0.815725 |
| 2026-10-22 ~20:30 | 13_retune_7days.csv | 0.817840 |
| 2026-10-22 ~22:05 | 15_avg_13_15.csv | 0.818070 |

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
- **Experiment 12 is the whole pipeline in one self-contained file**
  (`experiments/12_standalone/standalone.py`, no imports from other experiment folders).
  With `--init-std 0.05` it reproduces 11 exactly: identical cached features at cutoffs 5 and 7,
  fit-period MF AUC 0.78384, LightGBM 319 trees / 0.83360 on day 6. **Start new experiments
  from a copy of this file.**
- **Starting point of the item vectors** (fit period, MF AUC, day 4): co-click pre-training
  (PPMI-SVD) looked good with 1 seed (0.7915 vs 0.7838) but gave nothing with 3 (0.7944 vs
  0.7946): every seed starts from the same vectors, so it only removes seed noise. A smaller
  random start (std 0.005 instead of 0.05) kept a gain with 3 seeds (0.7973; 0.002/0.01: 0.7975/
  0.7969), but it shrinks as training data grows (+0.003 on day 2 → +0.0005 on day 6) and the
  final model tied with 11 (day-6 LightGBM 0.83359 vs 0.83360; public 0.81573 vs 0.81584).
- **Tune on as much data as the final model gets.** All MF settings had been tuned with 4 days
  of training data (days 0-3 → day 4); the submission MF trains on 7. Re-tuning on days 0-5 →
  day 6 (still fit-period data) found that SHORTER training is better even with 6 days: 3000
  steps instead of 5000 (MF AUC 0.8137 vs 0.8107 with 1 seed, 0.8175 vs 0.8148 with 3 seeds;
  2000/2500/4000/7000/9000 steps: 0.8114/0.8131/0.8128/0.8048/0.7982). l2 and click-log weight
  stayed. Day-6 LightGBM 0.8336 → 0.8353, public +0.002.
- **Tried on top of 13 and not kept** (experiment 14, fit period, days 0-5 → day 6):
  - Similarity features for LightGBM from the MF item vectors over the full history (max, top-3
    mean, mean, count of cosine > 0.5; 3 seeds): LightGBM on day 6 (split in halves by event)
    0.83283 vs 0.83263 without, noise.
  - Re-tuning at 3000 steps (1 seed, base 0.8137): l2 0.03 0.8141, learning rate 0.002 0.8141
    (noise); l2 0.08 0.8108, learning rate 0.005 0.8056, dim 64 0.8120, click-log weight
    0.05/0.2 0.8125/0.8097. The MF settings are saturated.
- **How the simulator behaves (checked on top of 13):**
  - No skip effect: candidates shown before and not clicked are clicked at 1.04x their item's
    usual rate (the raw 6.8% vs 5.9% was popularity). Repeat clicks are real: 4.1x.
  - No session/timing effects on "does the event get a click": time since the user's previous
    event, number of events so far: flat (0.62-0.64). Previous event clicked: 66% vs 57%, i.e. a
    stable per-user click propensity, already covered by user_rate: up-to-date user rate, event
    count and previous outcome as LightGBM features gave 0.83544 vs 0.83532 (day 6), noise.
  - Together with the earlier checks (no recency, no position, no multiple interests, no useful
    candidate-set signal): clicks behave like user taste (from the clicked items) x item vectors
    + item popularity + a per-user click propensity, which the current model already covers.
- **Tried on top of 13/15 and not kept:**
  - Smoothing item vectors over the co-click graph after training (v ← (1−a)v + a·neighbours,
    also rarity-weighted): always worse (days 0-5 → day 6 MF AUC 0.8137 unsmoothed; a = 0.1
    0.8127, 0.5 0.7891; rarity-weighted 0.802-0.810).
  - Impressions as a signal of item appeal: an item's click rate rises a little with how often
    it is shown (5.9% → 6.9% across impression quintiles), but impression shares are very stable
    (0.93 between halves of the week) and already in the features; evaluation-period impressions
    would add little.
  - Row weights 1/(user's events)^p so every user counts equally, as in evaluation.csv: first-
    event AUC on day 4 0.8054 (p=0), 0.8052 (0.5), 0.8048 (1).
  - XGBoost next to LightGBM (same 13 features; fit days 2-5, trees and weight picked on half
    of day 6, scored on the other half): LightGBM 0.83571, XGBoost depth 6 0.83584, rank
    average (w_xgb 0.7) 0.83589, i.e. +0.0002. Both are tree ensembles on the same features, so
    they make the same mistakes; not worth an upload on its own.
  - 17: training the MF on partly shortened histories (a share of training events cut to the
    k most recent clicks, k ≤ 20), aimed at light users. MF AUC on day 6 (3 seeds), all / users
    with 1-5 known clicks / first event of the day: none 0.8175 / 0.7525 / 0.8050; 20% cut
    0.8167 / 0.7513 / 0.8039; 40% 0.8160 / 0.7504 / 0.8031; 60% 0.8150 / 0.7500 / 0.8021; 40% cut
    to ≤ 5 clicks 0.8127 / 0.7471 / 0.7998. Worse in every group, light users included: their
    low AUC comes from having little information, not from a train/test mismatch.
  - Batch 1 (`lab/ease_poptaste.py`, half-B baseline 0.83571): "popularity taste" (mean
    log-popularity of the user's known clicks, and candidate minus that) 0.83595; EASE item-item
    scores (closed form, 15,000 most-clicked items = 52% of rows; full 50,000 items needs ~10 GB)
    0.83577-0.83585, with both 0.83591-0.83592. All noise. EASE alone scores only 0.61-0.63 on
    the rows it covers (MF 0.837 on the same rows).
  - Batch 2 (`lab/lgb_settings.py`, LightGBM settings re-checked on 13's features, half B):
    extra_trees 0.83565, 15 leaves / min_data 1000 0.83582, 255 leaves / min_data 500 0.83565,
    lambda_l2 30 + min_gain 1 0.83583, learning rate 0.02 0.83592 (811 trees). All noise; the
    LightGBM settings are saturated too.
  - MF size (`14_sim_features --tune`, days 0-5 → day 6, 3 seeds): dim 16 0.8147, dim 24 0.8171,
    dim 32 (current) 0.8175, dim 64 was 0.8120 (1 seed). 32 is the optimum.
  - 18: sequence model (transformer over the 50 most recent clicks, added to the MF's user
    vector, starting at zero): alone 0.8110 vs MF 0.8175; rank average with the MF 0.8175 vs
    0.8183 for the MF averaged with itself on other seeds. Click order carries nothing.
  - 19: items SHOWN in the user's earlier events as a second part of the user vector (learned
    weight starting at 0): 5 events 0.8173, 20 events 0.8172 vs 0.8175; users with 1-5 clicks
    0.7524 / 0.7523 vs 0.7525. The learned weight stays ~0.01: the candidate selection says
    nothing useful about the user beyond their clicks.
  - The three noise-level gains together (`lab/combined_small_gains.py`, half B): popularity
    taste + learning rate 0.02 0.83595, XGBoost with popularity taste 0.83599, rank blend of the
    two (XGBoost weight 0.6) 0.83605 = +0.0003 over 13. Below the +0.001 bar: no submission.
- **Where first events lose AUC** (validation cutoff 5, experiment 13, first events 0.8136):
  by the user's known clicks before the cutoff 0: 0.53 (1% of rows), 1-5: 0.75 (13%), 6-20:
  0.81 (46%), 21-100: 0.84 (40%). Later events look worse (0-6 h after the cutoff 0.825, 24-48 h
  0.797) only because they come from lighter users; within each group time has no effect.
  Candidates with 1-10 fit clicks 0.805 vs 0.830 for 11-100. Event level ("does the event get a
  click", sum of scores) only 0.67, within clicked events the click is ranked first 38% of the
  time.
- **First events are harder:** the same model scores 0.821 on all day-4 events but 0.805 on each
  user's first event of day 4 (validation of 13: 0.814). This matches the gap between day-6
  numbers and the leaderboard; it is a property of the evaluation events (lighter users, shorter
  histories), not of the training weights.
- Validation must mirror the official setup: 02 looked like 0.741 locally but its procedure
  leaked; the compliant version (03) scored 0.680 and tracked the leaderboard.

## Ideas not tried yet
- Empty-history events (~1% of rows) score ~0.55: a popularity-only fallback might help.
