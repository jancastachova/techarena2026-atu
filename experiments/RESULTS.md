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
| 15 | mean of 13 (MF seeds 0-14) and a twin run with MF seeds 15-29 (`15_seeds15_29.csv`, day-6 LightGBM 0.8355) | not run ‡ | 0.81807 | `15_avg_13_15.csv` |
| 20 | 13 with the user vector = sum / n^0.9 instead of the mean (MF +0.002, LightGBM absorbs it) | not run ‡ | 0.81761 | `20_user_vector.csv` |
| 24 | rank average of LightGBM + XGBoost on MF30 + popularity taste + the standalone sequence model's features (18 `--pure-seq`, 1 seed), and an MLP; half B 0.83684 vs 0.83612 | not run ‡ | 0.81853 | `24_ensemble_seq1.csv` |
| 24 | the same with the sequence features averaged over 4 seeds and the MLP also given them; half B 0.83701 | not run ‡ | 0.81891 | `24_ensemble_seq4.csv` |
| 25 | 24 (seq4) + an IDF-pooled MF (25, user vector weights (1+clicks)^-1, 8 seeds) as features for all three members; half B 0.83773 | not run ‡ | 0.81965 | `25_ensemble_idf.csv` |
| 25 | the same with 8 sequence seeds (4 with 1 layer) and IDF gammas 1.0 / 1.5 / 2.0 as three feature groups; half B 0.83796 | not run ‡ | 0.81983 | `25_ensemble_seq8_idf3.csv` |
| 25 | + a clicks-only MF (25 `--impr-weight 0 --aux-weight 1`, 8 seeds) as a fourth feature group; half B 0.83850 | not run ‡ | 0.82001 | `25_ensemble_clicks.csv` |
| 25 | + 12 sequence seeds (4 more, 1 layer, trained on the Apple GPU) and an impressions-only MF (`--aux-weight 0`); half B 0.83883 | not run ‡ | **0.82033** | `25_ensemble_impronly.csv` |

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
| 2026-10-24 06:42 | 20_user_vector.csv | 0.817614 |
| 2026-10-24 09:27 | 24_ensemble_seq1.csv | 0.818526 |
| 2026-10-24 11:40 | 24_ensemble_seq4.csv | 0.818910 |
| 2026-10-24 ~13:36 | 25_ensemble_idf.csv | 0.819645 |
| 2026-10-24 14:27 | 25_ensemble_seq8_idf3.csv | 0.819829 |
| 2026-10-24 15:09-15:25 | 25_ensemble_clicks.csv | 0.820010 |
| 2026-10-24 17:30-17:32 | 25_ensemble_impronly.csv | 0.820330 |

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
  - A neural network (3-layer MLP on LightGBM's features, Gaussian-rank inputs, 3 seeds;
    `lab/mlp_blend.py`): alone 0.83579 vs LightGBM 0.83571, rank correlation with LightGBM
    0.995, best blend 0.83603 (+0.0003). Two different model families reach the same AUC from
    the same features: the SECOND stage is saturated given these features.
  - 20: how the user vector is normalised, `sum(v_h) / n^alpha` (13 uses alpha = 1, a plain
    mean). MF alone, days 0-5 → day 6, 3 seeds: alpha 0.5 0.7970, 0.75 0.8161, 0.85 0.8186,
    **0.9 0.8189**, 0.95 0.8185, 1.0 0.8175, 1.05 0.8158; a learned dim x dim map between the
    history and candidate roles (bilinear) 0.8123, so tying was right. At 15 seeds alpha 0.9
    improves the MF on EVERY fit day (day 2 0.78125 → 0.78228, day 6 0.81891 → 0.82087, +0.002).
    **But LightGBM gives it all back** (`lab/mf_compare.py`, half B): MF A (13) 0.83571, MF B
    (alpha 0.9) 0.83550, both feature sets together 0.83568, rank blend +0.0001; the two
    pipelines' predictions correlate 0.9966. `submissions/20_user_vector.csv` was written but
    is not worth an upload.
- **The bottleneck is the path between the stages, not either stage.** MF-level gains have now
  failed to reach the final score four times (08 choice loss, 10+11 MFs together, 18 sequence,
  20 alpha), and a different second-stage model family (MLP) ties LightGBM on the same features.
  The base features already encode what the MF gains. What is untested is base features that do
  NOT come from the MF.
  - Item x time-bucket click rate (`lab/item_time_ctr.py`): base features hold the item's
    overall rate and the 6-hour bucket separately, never their interaction. Items with >= 30
    impressions in every bucket show a median max/min bucket rate of 1.99, but that spread is
    noise: the same items' bucket rates correlate 0.053 between the days before day 6 and day 6
    itself. Half B 0.83572 vs 0.83571. (Impressions per bucket are very uneven: 10/20/30/40%.)
  - A separate event-level model (`lab/two_stage.py`): the global AUC splits into "which events
    get a click" and "which candidate inside the event", and the first-event diagnosis found the
    event part weak (0.67) with a +0.038 counterfactual. A LightGBM trained on one row per event
    (label = any click), with the spread of the candidate scores as extra features, reaches only
    **0.69123 on half B vs 0.68989 for the sum of the row model's scores**: the row model already
    extracts almost all of the event-level signal, and the counterfactual was unreachable, not
    untried. Combining: within-event share x event probability 0.83600 (+0.0003), row log-odds +
    0.1 x event log-odds 0.83530 (worse).
  - Equal-weight ensemble of everything that gave ~+0.0003 (`lab/final_ensemble.py`, half B,
    weights fixed in advance so nothing is tuned): members LGB 13 0.83571, LGB + popularity
    taste at lr 0.02 0.83595, LGB on experiment 20's MF 0.83550, XGBoost + popularity taste
    0.83599, MLP 0.83579. Rank averages: 2 members 0.83591, 3 0.83595, 4 0.83603,
    **5 0.83613 (+0.00042)**. The gain grows with diversity but every member correlates
    >= 0.9947 with LGB 13, so it plateaus far below the +0.001 bar; ~+0.0001 per extra member
    for about an hour of compute each.
  - 21: a LEARNED NONLINEAR interaction instead of the dot product -- MLP([hv, q, hv*q]) added
    to the logit, zero-initialised so training starts at 13 (the NeuMF / DeepFM / DCN idea, the
    standard modern CTR architecture). MF alone, days 0-5 → day 6, 3 seeds: **0.78908 vs 0.81750**;
    hidden 128 0.78767, dropout 0.1 0.78695. Much worse, at every history length. The simulator
    generates clicks through a bilinear mechanism (taste . item vectors + popularity), so extra
    expressive power has nothing real to fit and memorises instead. Together with the MLP and
    XGBoost results: **a more powerful model is the wrong axis here.**
  - 22: the MF's 32 element-wise products hv*q as LightGBM features instead of only their sum
    (`lab/product_dims.py`; vectors exported by `22_product_dims/product_dims.py --save-vectors`,
    1 seed, since dimensions are not comparable across seeds). Half B: 13 0.83571, + 32 product
    dims 0.83562, + product dims + the candidate's own 32 dims 0.83566. **LightGBM ignores them**
    -- none of the 32 appears in its top 12 features by gain (those stay mf_logit_share,
    mf_hist_dot, mf_logit). The dot product is a sufficient summary of the 32 dimensions, so the
    compression between the stages is NOT why MF gains fail to propagate.
  - Fitted stacking instead of equal weights (`lab/final_ensemble.py`, logistic combiner fitted
    on half A, read on half B): log-odds 0.83619, ranks 0.83619 (+0.00048) vs equal weights
    0.83613 (+0.00042); stable across l2 1 and 100. The fitted weights rank the members by how
    much they DIFFER from 13 (MLP +0.69, MF alpha 0.9 +0.36, XGBoost +0.55, LGB 13 slightly
    negative), which confirms diversity is what pays -- and that we have run out of diverse
    members, not that the combiner was wrong.
- **Loss and sampling, for the record** (often suggested, already covered): the metric is GLOBAL
  AUC over all rows, not per-event NDCG/MRR, so event-grouped ranking losses are misaligned and
  measured worse (lambdarank 0.7808, rank_xendcg 0.7802 vs 0.7863 pointwise; the listwise choice
  model of 08 improved the MF but tied after LightGBM). Negatives are NOT uniform: the click-log
  loss is a sampled softmax against 9 impression-weighted negatives (49 negatives: worse), and
  the main loss uses every real impression. Item-item transitions P(B|A) (`trans_decay`,
  `trans_last`), co-click similarity and time-since-last-action are already features. Hard
  negatives do not exist here: candidates shown before and not clicked are clicked at 1.04x
  their item's normal rate.
- **Evaluation-period impressions, now measured** (`lab/window_impressions.py`; RESULTS.md had
  dismissed this on judgement). Counting how often an item appears in the candidate sets uses no
  labels, so it is a legal popularity refresh after the cutoff freeze. Impression shares correlate
  0.948 between periods, drift is small (10th-90th percentile 0.67x-1.38x), and the change in
  share correlates **+0.0044** with the day-6 click residual, i.e. nothing. As features
  (in-window impressions, share, change in share): half B 0.83581 vs 0.83571, and they rank
  19th/21st/29th by gain. Popularity drift carries no click information.
- **The model is already calibrated across segments** (`lab/segment_calibration.py`). Global AUC
  compares rows from different segments, so a mutual offset between them would cost AUC for free.
  Measured on half A (mean score / actual click rate): 0 known clicks 1.009, 1-5 0.982, 6-20
  0.998, 21+ 1.000; the log-odds shifts needed are +0.000, +0.020, +0.002, -0.000, and applying
  them changes half B by less than 0.00001. Per-segment binned recalibration is WORSE (10 bins
  0.82988, 25 0.83449, 50 0.83506, 100 0.83470 vs 0.83571): the bins destroy real within-segment
  ordering. LightGBM calibrates itself from the history-length and user-rate features.
- **Rare items are UNDERFIT, not overfit** (23, frequency-adaptive l2: item i's penalty scaled by
  (1 + K/(1+clicks_i))^beta, so rare items are shrunk toward "popularity only"). MF alone,
  days 0-5 → day 6, 3 seeds, all / candidates with 1-9 clicks (15% of rows): uniform (= 13)
  0.81750 / 0.76531; beta 0.5 0.81648 / 0.76183; beta 1.0 0.81464 / 0.75529; beta 2.0 0.80677 /
  0.72618; beta 1.0 with K 50 0.79502 / 0.70075. Shrinking rare items hurts **them** most, so
  a vector built from 3-9 clicks carries real signal and shrinking destroys it. Their low AUC is
  a lack of information, not fitted noise -- which is why five global l2 tunes all agreed.
- **More training rows do not help LightGBM** (`lab/early_days.py`): days 0-1 are dropped because
  no MF can score them; adding them with the MF features marked missing (5.89M rows instead of
  3.92M, +50%) gives half B 0.83565 (days 0-5) and 0.83568 (days 1-5) vs 0.83571 (days 2-5).
  LightGBM is not data-limited at 3.9M rows.
- **Ties from writing 6 significant digits cost nothing.** `15_avg_13_15.csv` has 435,270
  distinct values for 529,390 rows (170,718 rows in tie groups). On day 6 half B, rounding
  LightGBM's predictions to 6 significant digits cuts 489,760 distinct values to 417,349 but
  leaves the AUC identical to six decimals (0.835713); breaking ties with mf_logit changes
  nothing either. The tied rows differ only in the 7th digit.
- **The true baseline is the 13 + 15 average, not 13 alone.** `lab/ensemble_v2.py` (half B):
  LightGBM on 13's MF 0.83571, on 15's MF 0.83585, their rank average (= the current best
  submission) **0.83612**. LightGBM on the MF averaged over all 30 seeds 0.83603; + popularity
  taste at lr 0.02 0.83625; XGBoost + popularity taste 0.83611; MLP (MF30 + popularity taste)
  0.83629, the best single member; 5 members on MF30, equal weights, 0.83642 (+0.00030).
- **The standalone sequence model adds real information.** Experiment 18 with `--pure-seq` (user
  vector = transformer over the raw item-ID sequence ALONE, no mean-of-clicks anchor) scores
  0.791 on day 6 with 1 seed (MF 0.819) but its ranks correlate only **0.860** with the MF's;
  every other member sits at 0.995+. As LightGBM features (its 4 scores + event-relative
  versions; `lab/seq_feature.py`): 0.83617 vs 0.83571 on 13; as a plain blend only +0.00014.
  On the best members (`lab/ensemble_v3.py`, vs the true baseline 0.83612): LightGBM MF30 +
  popularity + sequence (lr 0.02) 0.83664, XGBoost with the same features 0.83665, and their
  rank average with the MLP **0.83684 (+0.00073)**. Adding 13's and 15's LightGBMs back in
  dilutes it (0.83668). Built as `24_ensemble/ensemble.py --members lgb_popseq,xgb_popseq,mlp_pop`
  (1 sequence seed): `submissions/24_ensemble_seq1.csv`. Three more sequence seeds are
  running for a less noisy version.
- **Diversity alone is not enough** (`lab/extra_member.py`). Experiment 21's deep interaction as
  a full run (`--deep --mf-seeds 5`) scores 0.797 on day 6 and is the most different model yet
  (rank corr 0.895 with MF30, **0.781 with the sequence model**), but added as LightGBM features
  on top of MF30 + popularity + sequence it gives half B 0.83670 vs 0.83664 (its features rank
  15th-21st by gain). The sequence model is diverse AND informative; the deep model is only
  differently noisy.
- **Four sequence seeds** (`lab/ensemble_v4.py`; 18 `--pure-seq --mf-seeds 1 --seed-offset 0..3`,
  day-6 AUC per seed 0.791 / 0.775 / 0.784 / 0.783): the average scores **0.807** alone but
  correlates 0.931 with MF30 (one seed: 0.860) -- averaging removes noise, and part of a noisy
  model's disagreement WAS noise. Half B vs the true baseline 0.83612: LightGBM MF30 + popularity
  + sequence 0.83672 (1 seed: 0.83664), XGBoost 0.83665, MLP MF30 + popularity + sequence
  **0.83683** (best single member); rank average of LightGBM + XGBoost + MLP, all with sequence
  features, **0.83701 (+0.00089; +0.00017 over the uploaded 24_ensemble_seq1 combination,
  0.83684)**. Built as `24_ensemble_seq4.csv`: **public 0.81891** (+0.00038 over the 1-seed
  version, about twice the half-B gain -- the sequence features help first events more than
  day 6 as a whole, so half B UNDER-states sequence gains).
- **Sequence model settings** (18 `--tune pure,...`, MF-style day-6 check, 1 seed each): 50
  clicks and 2 layers 0.79092 (21 min per day-model); 100 clicks 0.78669 (40 min); **1 layer
  0.79492 (10 min)**. One seed is noisy (four 2-layer seeds: 0.775-0.791), but 1 layer is at
  least as good at half the cost, so new seeds use it (`--seq-layers 1`). The same seed with a
  different architecture correlates only 0.83-0.88, so mixing architectures adds diversity.
- **Weighted pooling of the clicked items** (25, `sum(w_h v_h) / sum(w_h)`, MF-style day-6 check,
  3 seeds): mean 0.81750; idf w = (1+clicks)^-0.5 0.81774 (rank corr with the mean 0.988);
  idf ^-1.0 0.81316 (corr 0.954); learned per-item weights 0.81003 (overfit). idf 1.0 is being
  tested as an extra member next to the sequence model.
- **IDF pooling is a second source of new information.** Full run of 25 with `--pool idf
  --pool-gamma 1.0 --mf-seeds 8`: day 6 0.81529 alone (MF30 0.819), rank corr 0.958 with MF30
  and 0.918 with the sequence model. Added as LightGBM features (`lab/extra_member.py`) on top
  of MF30 + popularity + 4-seed sequence: half B **0.83672 -> 0.83748 (+0.00076; +0.00136 vs
  the true baseline 0.83612)**, its features ranking 4th, 7th and 9th by gain. A click on a
  niche item says more about taste than one on an item everybody clicks, and nothing else in
  the pipeline could express that.
  Full ensemble (`lab/ensemble_v5.py`, half B; LightGBM lr 0.02 / XGBoost / MLP each on MF30 +
  popularity + 4-seed sequence + IDF 1.0): 0.83748 / 0.83732 / 0.83750, rank average
  **0.83773 (+0.00161 vs the true baseline, +0.00072 vs the uploaded 24_ensemble_seq4
  combination 0.83701)**; half A 0.83714 vs 0.83621. Built as `25_ensemble_idf.csv`
  (`24_ensemble/ensemble.py --members lgb_full,xgb_full,mlp_full --extra idf=...`):
  **public 0.81965** (+0.00074, matching the half-B gain almost exactly).
- **More sequence seeds and stronger IDF views add up** (`lab/ensemble_v5.py`): 8 sequence
  seeds (four 2-layer + four 1-layer, `--seq-layers 1` at half the cost) and three IDF groups
  (gamma 1.0 / 1.5 / 2.0; alone 0.815 / 0.807 / 0.795 on day 6). Half B: sequence-only LightGBM
  0.83691 (4 seeds 0.83672); LightGBM / XGBoost / MLP full 0.83767 / 0.83748 / 0.83764; rank
  average **0.83796, half A 0.83744** (vs 0.83773 / 0.83714 for 25_ensemble_idf). Built as
  `25_ensemble_seq8_idf3.csv`: **public 0.81983** (+0.00018, about 0.8x the half-B gain).
- **A clicks-only MF is a third real view; anti-IDF is not** (`lab/extra_member.py --base-extra`,
  on top of MF30 + popularity + 8 sequence seeds + 3 IDF groups, LightGBM half B 0.83767).
  Clicks-only = 25 with `--impr-weight 0 --aux-weight 1` (item vectors learned only from which
  items the same users click; 0.792 alone, rank corr 0.885 with MF30 and **0.810 with the
  sequence model**): **0.83813 (+0.00046)**, its in-event share ranking 5th by gain. Anti-IDF
  (weights (1+clicks)^+0.5, popular clicks count more; 0.813 alone, corr 0.987 with MF30):
  0.83753, worse -- the same information as the MF.
  Full ensemble with clicks-only added to all three members (`lab/ensemble_v5.py`, half B, vs
  25_ensemble_seq8_idf3's combination): LightGBM 0.83767 -> 0.83813, XGBoost 0.83748 -> 0.83797,
  MLP 0.83764 -> **0.83835**, rank average **0.83796 -> 0.83850 (+0.00054)**, half A 0.83744 ->
  0.83802. Built as `25_ensemble_clicks.csv`: **public 0.82001** (+0.00018, about 0.3x the half-B
  gain). Half-B-to-public conversion over the day's five uploads: 0.6x, 2.2x, 1.0x, 0.8x, 0.3x --
  about 1x on average, but single steps of this size are partly leaderboard noise.
- **Diminishing returns from more views** (`lab/extra_member.py`, LightGBM half B on top of MF30 +
  popularity + sequence + 3 IDF + clicks-only): 12 sequence seeds instead of 8 0.83813 -> 0.83827;
  then + event-choice MF (25 `--loss choice`: softmax over the 10 candidates + a no-click option;
  0.810 alone, the strongest single alternative model) 0.83831; + impressions-only (`--aux-weight
  0`) 0.83842; + clicks-only with IDF 0.83835. The choice view agrees 0.94-0.95 with the views
  already in, so it adds almost nothing despite its strength.
  Full ensemble with 12 sequence seeds + impressions-only (`lab/ensemble_v5.py`, half B, vs the
  25_ensemble_clicks combination 0.83850): LightGBM 0.83842, XGBoost 0.83831, MLP 0.83867, rank
  average **0.83883 (+0.00033)**, half A 0.83829 (+0.00027). Built as `25_ensemble_impronly.csv`:
  **public 0.82033** (+0.00032, matching the half-B gain).
  Not kept: clicks-only with UNIFORM click-log negatives (25 `--neg uniform`, 0.752 alone):
  0.83838 vs 0.83842.
- **Denoising the core MF hurts** (25 `--tune hdrop:...`: each known click hidden from the user
  vector with probability p during training; MF alone, days 0-5 -> day 6, 3 seeds): p = 0 0.81750,
  0.2 0.81668, 0.4 0.81550, 0.6 0.81270, 0.4 with 5000 steps 0.81308. Like the shortened histories
  of 17: every known click carries information the model needs.
- **The stage-2 MLP is near its best** (`lab/mlp_sweep.py`, Apple GPU, full current feature set,
  3 seeds, half B): 256-128-64 / drop 0.1 / lr 1e-3 (current) 0.83860, 512-256-128 / drop 0.2
  0.83863, 512-256-128-64 0.83872, current with lr 3e-4 0.83887, 128-64 0.83864; the deeper net
  with 10 seeds 0.83886. About +0.0002 for the MLP alone, less in the ensemble.
- **Item click rate within the user's taste group does not help** (`lab/segment_ctr.py`: users
  grouped by k-means on their taste vector, k = 10 / 30, groups and counts as of each day; the
  classic item x user-segment target encoding of CTR competitions). Alone 0.635 (lift 0.647); on
  top of the current best feature set 0.83827 vs 0.83842, ranked 45th-61st by gain. With ~140
  impressions per item, a group has too few to beat the learned models' estimates.
- **Co-impressions carry no taste signal** (`lab/coimpression.py`: item vectors from positive PMI
  of "shown in the same event", SVD dim 32, as of each day). Aimed at rare items (candidates with
  <= 10 clicks are 66% of rows). Alone 0.580, rare candidates 0.542; on top of the current best
  0.83854 vs 0.83842, rare candidates 0.79546 -> 0.79550. Candidates are picked almost
  independently of taste (cf. cosine 0.034 vs -0.001), so co-shown items share no audience.
- **Half-B -> leaderboard conversion differs by change**: more sequence seeds 2.2x (1 -> 4 seeds),
  clicks-only 0.3x, IDF 1.0x. The sequence features seem to help first events more than day 6 as
  a whole; more sequence seeds are running on the Apple GPU.
- **What the best model still gets wrong is (almost) noise** (`lab/residuals.py`). Day-6 residuals
  split into two halves by event: per item, half-to-half correlation +0.021 (+-0.011) and residual
  variance equal to binomial noise (1.01); per user +0.052 (+-0.016) -- but across DAYS (trained on
  days 2-4, day 5 vs day 6, `--cross-day`) only +0.015 (+-0.012): the user effect is same-day
  "mood", which the first-event test cannot use. Calibration by known clicks, time bucket and
  MF rank is 0.98-1.03 everywhere except users with no known clicks (1.12, 0.8% of rows).
- **No hidden genres in the simulator** (`lab/simulator_categories.py`). k-means on the MF's item
  vectors; a heavy user's share of clicks in their top cluster vs a shuffled baseline with the
  same cluster sizes: k=5 0.466 vs 0.323, k=20 0.270 vs 0.169 (1.59x, the peak), k=100 0.161 vs
  0.115. Discrete genres would give 0.6-0.9; taste is continuous and spread out, which is what
  the 32-dim MF assumes. But candidates from a user's favourite cluster (k=20, 6.1% of
  candidates vs 5.0% by size) are clicked 28% of the time vs 5% otherwise. As a coarse "cluster
  affinity" feature (`lab/cluster_affinity.py`: smoothed share of the user's known clicks in the
  candidate's cluster, clusters refitted per day on MF vectors trained only on earlier days,
  k = 20/50/200): click rate 3.3% -> 16.0% across lift bins, but on top of the current best
  feature set 0.83832 vs 0.83842 (ranks 28-38 by gain). The views already carry it.
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
