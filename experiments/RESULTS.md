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
| 25 | + 12 sequence seeds (4 more, 1 layer, trained on the Apple GPU) and an impressions-only MF (`--aux-weight 0`); half B 0.83883 | not run ‡ | 0.82033 | `25_ensemble_impronly.csv` |
| 26 | 25 (impronly) with 16-seed IDF and clicks-only views + an AUDIENCE view (item scored by the mean taste of its past clickers minus the mean taste of users it was shown to, 5 MF vector seeds); half B 0.84025 | not run ‡ | 0.82142 | `26_ensemble_audience5.csv` |
| 26 | + two more hops of the audience view (hop2c: the user's mean audience-minus-exposure over their clicks vs the candidate's; hop3: the audience operator applied twice); half B 0.84122 | not run ‡ | 0.82235 | `26_ensemble_hops.csv` |
| 26 | + four views chosen on the evaluation-weighted half B (hop2, idf_hop2, uoff10, dim64); half B 0.84134, eval-weighted 0.82466 (+0.00034 [+0.00013, +0.00054]) | not run ‡ | 0.82268 | `26_ensemble_combo4.csv` |
| 26 | + five more views (choice loss, clicks-only IDF, clicks-only uniform negatives, user vector sum/n^0.9, deep interaction); eval-weighted vs 26_ensemble_hops +0.00066 [+0.00041, +0.00088] | not run ‡ | 0.82271 | `26_ensemble_bundle.csv` |
| 26 | the bundle averaged over two model seeds (`24_ensemble/ensemble.py --model-seed 0` and `1`) | not run ‡ | 0.82277 | `26_ensemble_bundle_2seeds.csv` |
| 26 | mean of four bundle builds: model seeds 0 and 1, each with the 30-seed and the 45-seed core MF (`--mf-core-extra`) | not run ‡ | 0.82279 | `26_ensemble_bundle_4builds.csv` |
| 34 | bundle + sqrt-set view + COSINE view + eight cosine audience views (`34_cosine/build.sh 0`); eval-weighted half B +0.00125 [+0.00085, +0.00173] vs the bundle set | not run ‡ | 0.82389 | `34_cosine_s0.csv` |
| 35 | 34 + PROJECTED cosine view (8 strong principal directions) + cosine audience with unit clicker tastes (`35_projection/build.sh 0`); eval-weighted half B 0.82569 -> 0.82718 | not run ‡ | 0.82520 | `35_projection_s0.csv` |
| 35 s1 | 35 with model seed 1 (`35_projection/build.sh 1`); public seed-to-seed gap 0.00004 | not run ‡ | 0.82516 | `35_projection_s1.csv` |
| 35b | mean of 35 seed 0 and seed 1 builds (`35_projection/build.sh 1`; Spearman 0.9992 between seeds) | not run ‡ | **0.82522** | `35_projection_2seeds.csv` |
| 36 | 35 set + temp06, cosresid8, cossoft20, soft_aud views; binary LightGBM replaced by lambdarank over random 1000-event groups (`36_combo/build.sh 0`, members lgbrank/xgb/mlp, 6.5 h for the ranking member); lab paired vs 35 binary half B +0.00036, eval-weighted 0.82718 -> 0.82777 (+0.00059). Public +0.00006 over 35 s0, +0.00003 over 35b: most of the lab gain did not transfer (eval-weighted to public gap 0.0020 -> 0.0025) | not run ‡ | **0.82526** | `36_combo_s0.csv` |
| 36b | equal mean of 36 s0 and 35b, weights fixed in advance (`15_more_seeds/average.py`; Spearman 36 vs 35b 0.9986, 35's seed pair 0.9992). +0.00010 over 35b, +0.00007 over 36: diversity between builds that differ in features or objective pays more than the ranking member's own gain. Superseded as final pick by 37c | not run ‡ | **0.82532** | `36b_avg_36_35b.csv` |
| 37 | SET36 with four members: binary LightGBM (1558 trees, day-6 0.84403), lambdarank LightGBM over random 500-event groups (2200 fixed trees, 1.6 h), XGBoost (0.84385), MLP; rank-averaged (`37_rank4/build.sh 0`, 3.0 h). Day-5 fresh holdout of the change: +0.00050 eval-weighted, first events +0.00047. Spearman vs 36b 0.9996 | not run ‡ | not uploaded | `37_rank4_s0.csv` |
| 37b | equal mean of 37 s0 and 36b, weights fixed in advance (`15_more_seeds/average.py`) | not run ‡ | 0.82532 | `37b_avg_37_36b.csv` |
| 37x2 | mean of 37 seed 0 and seed 1 builds (`37_rank4/build.sh 1`: lgb 0.84398, xgb 0.84392; Spearman 0.9994 between seeds) | not run ‡ | 0.82534 | `37_rank4_2seeds.csv` |
| 37c | equal mean of 37x2 and 36b, weights fixed in advance (same recipe as 36b; Spearman 37x2 vs 36b 0.9996). +0.00003 over 36b, +0.00002 over 37x2: within public noise, the 37 builds add little over 36b. Superseded as final pick by 38b | not run ‡ | **0.82535** | `37c_avg_37x2_36b.csv` |
| 38 | SET37 with cosproj8 rebuilt from aux-0.3/4500-step MF vectors (10 seeds) + activity bundle act_a/b/c (`38_aux_proj/build.sh 0`, 4 h): lgb day-6 0.84426 (37: 0.84403), xgb 0.84432 (0.84385). Lab vs SET36: day 6 +0.00037 plain / +0.00041 eval-weighted, day-5 +0.00045 / +0.00050. Spearman vs 37c 0.9983. Public 0.00010 below 37c alone, but single builds run below averages (36 s0 0.82526, 35 s0 0.82520) | not run ‡ | 0.82525 | `38_aux_act_s0.csv` |
| 38b | equal mean of 38 s0 and 37c, weights fixed in advance (`15_more_seeds/average.py`). +0.00005 over 37c: within public noise. Superseded as final pick by 39b | not run ‡ | **0.82540** | `38b_avg_38_37c.csv` |
| 39 | neural CHOICE model member alone (`39_choice/build.sh 0`: one softmax per event over 10 candidates + no click, MLPs on the lean set SETLEAN, 3 seeds, 7/7/3 epochs by early stopping on day 6 (day-6 AUC 0.84438 / 0.84438 / 0.84405), refit on days 2-6; 10 min). Lab day 6 alone 0.84508 / eval-weighted 0.82833 vs lean LightGBM 0.84447 / 0.82779. Spearman vs 38b 0.9944 (the most different strong build so far). Public below 38 alone (0.82525) although the lab put it above LightGBM: the choice model's lab edge does not transfer as a single model | not run ‡ | 0.82509 | `39_choice_s0.csv` |
| 39b | equal mean of 39 and 38b, weights fixed in advance (lab: equal rank blend of choice model and LightGBM +0.0005-0.0006 on days 6 and 5). +0.00014 over 38b, the largest step since 36b; diversity pays. Superseded by 40b | not run ‡ | **0.82555** | `39b_avg_39_38b.csv` |
| 40 | lean set (15 groups) with the projected view from 9000-step vectors (`40_long_mf/build.sh 0`): binary LightGBM (1558 trees, day-6 0.84464), XGBoost (635 trees, 0.84457), choice model (3 seeds); rank-averaged, no lambdarank. Lab vs lean set +0.0006 eval-weighted on days 6 and 5. Public +0.00041 over 38 s0 (0.82525) -- exactly the clean day-4 prediction for SET38 -> SET40 (+0.00041): the best single build so far and the first lab gain since 35 to transfer in full | not run ‡ | 0.82566 | `40_long_mf_s0.csv` |
| 40b | equal mean of 40 and 39b, weights fixed in advance. +0.00021 over 39b at half weight: the 40 change transfers (day 4 predicted ~+0.0004 for 40 alone over 38) | not run ‡ | **0.82575** | `40b_avg_40_39b.csv` |
| 42 | 40 + a shallow regularised LightGBM member (15 leaves, min_data_in_leaf 2000, feature_fraction 0.3; `42_shallow/build.sh 0`, 3401 trees, day-6 0.84474, 10 min) as a 4th equal rank member (40 x3 + member x1). Lab 50/50 with the SET40 LightGBM, gap-weighted: day 6 +0.00010, day 5 +0.00016, day 4 (unselected) +0.00031. Spearman member vs 40 0.9981 | not run ‡ | not uploaded | `42_shallow_s0.csv` |
| 42b | equal mean of 42 and 39b (as 40b). Spearman vs 40b 0.99997. Public +0.00001 over 40b: as expected for a 12.5%-weight member, within noise; kept as the final pick on its 3-day lab evidence | not run ‡ | **0.82576** | `42b_avg_42_39b.csv` |
| 43s3 | 42's four members with the CHOICE member at 0.40 (others 0.20; `43_choice_weight/mix_choice.py`: 0.8 x the equal 4-member file + 0.2 x rank(choice member)), averaged over model seeds 0-2 (seeds 1-2 built on the vast.ai box). Lab choice 0.35-0.45 beats 0.25 on days 6, 5, 4; whole ensembles on the never-used day 3: 42 config -> this +0.00013 eval / +0.00018 gap-weighted, intervals above 0 | not run ‡ | not uploaded | `43s3_choice04_3seeds.csv` |
| 43s3b | equal mean of 43s3 and 39b (the 40b / 42b recipe). Public +0.000009 over 42b: in the right direction, within noise (the choice-weight change moves ~8% of the final weight; Spearman vs 42b ~0.9999) | not run ‡ | **0.82577** | `43s3b_avg_43s3_39b.csv` |
| 45s3 | 43s3 with the soft-projected AUDIENCE view rebuilt from the 40 vectors (soft_aud <- 40c_aud; `45_aud40/build.sh`, 3 model seeds on the box, choice member on the GPU, mixed to choice 0.40). Representation change: lab paired vs SET40 day 6 / 5 / 4 eval-weighted +0.00004 / +0.00023 / +0.00033, gap-weighted +0.00021 / +0.00028 / +0.00023. Spearman vs 43s3 0.99909. Public +0.00030 over 43s3b: the representation gain transferred ~1:1 | not run ‡ | **0.82607** | `45s3_aud40_choice04.csv` |
| 45s6 | 45s3 with 3 more build seeds (3-5, box, `45_aud40/build.sh "3 4 5"`): each seed mixed to choice 0.40, equal mean of 6. Variance reduction for the private split, no new information. Spearman vs 45s3 0.99996 (a single seed vs 45s3: 0.99968). Public -0.000006 vs 45s3: same model, as expected | not run ‡ | 0.82606 | `45s6_aud40_choice04.csv` |
| 45s3b | equal mean of 45s3 and 39b (the recipe of 40b / 42b / 43s3b; halves the new view's weight). Spearman vs 43s3b 0.99977 | not run ‡ | not uploaded | `45s3b_avg_45s3_39b.csv` |
| 47s6 | 45's trees with TWO choice members replacing the 45 choice: SET45 + SPECTRAL taste view (`lab/spectral_view.py`, learned per-direction metric) and D45 (337 features), 0.25 each, trees 1/6 each (`47_spec_choice/build.sh`, choice members on the box, model seeds 0-5). Lab vs the 45 composition, eval- / gap-weighted: day 6 +0.00019 / +0.00017, day 5 +0.00013 / +0.00011, day 4 +0.00002 / +0.00003, day 3 (never used) +0.00005 / +0.00009 (choice 0.4; 0.5 adds +2..+7 x1e-5 on every day). Spearman vs 45s6 0.99972 | not run ‡ | **0.826168** (+0.000099 vs 45s3) | `47s6_spec_choice.csv` |
| 48 | 47s6 with a LightGBM LAMBDARANK member (random 500-event groups, 2200 fixed trees, SET45, seed 0; fourth box `tools/box2_rank.sh`): trees 0.125 each, lambdarank 0.125, choice SP + D 0.25 each (`experiments/48_rank_member/build.py`; rebuild without the member reproduces 47s6 to 1e-10). Day-6 check on SET36 members (`lab/combine_check.py`): +0.00004 [+0.00001, +0.00006]. On SET45 (`tools/box2_check48.sh`): day 6 +0.00004 [+0.00001, +0.00007], day 5 +0.00005 [+0.00001, +0.00009]. Spearman vs 47s6 0.99998 | not run | 0.826164 (-0.000004 vs 47s6: the lambdarank member does not carry to the leaderboard) | `48_rank_member.csv` |
| 49 | BAYESIAN CHOICE families F1 + F2 (`49_bayes_choice/bayes.py`, K=8, ALS start, fixed priors, 4 seeds) with existing choice + ensemble in a conditional-logit stacker with an outside option (`stack.py`, `final.py`). Jan24 +0.00281, Jan25 purged confirmation day-5 rows +0.00341, all rows +0.00309, 5/5 user folds | not run ‡ | **0.82904** | `49_bayes_stack_s4.csv` |
| 49s8 | 49 with 8 seeds of F1 and F2 (Jan24 +0.00020 over 4 seeds, 5/5 folds; public +0.00020) | not run ‡ | **0.82924** | `49_bayes_stack_s8.csv` |
| 49c | 49s8 + a category tree member as a fifth stacker input (`final2.py`; Jan24 +0.00086, 5/5 folds; public +0.00054, about 60% of the lab gain: the evaluation members are stronger than the lab ones, so more overlaps) | not run ‡ | **0.82978** | `49c_bayes_cat_s8.csv` |
| 49d | 49c with 16 seeds + choice member with category columns + two point-estimate families (c16, gau; picked on Jan23, tested once on Jan24), `final3.py`. Jan24 +0.00083 in total; public +0.00122 | not run ‡ | **0.83100** | `49d_stack.csv` |
| 49e | 49d + tree and choice members trained WITH the point-estimate view as features (`build_view.py`, `final4.py`; each fit day from a model of earlier days). Alone each beats the 49d stack on Jan24; together +0.00104, 5/5; public +0.00069 | not run ‡ | **0.83169** | `49e_view.csv` |
| 49f | 49e + tree and choice members with the Bayes chains (F1, F2, 4 seeds) as a second view (`build_view2.py`, `final5.py`). Jan24 +0.00039, 5/5; public +0.00016. All-views choice member +0.00009 on Jan24: rejected | not run ‡ | **0.83185** | `49f_view2.csv` |
| 49f2 | 49f with the view tree averaged over two model seeds; best public, leaderboard 24th of 76 | not run ‡ | **0.83187** | `49f2_tree2.csv` |
| 49g | 49f2 + a choice member with the Bayes view at 6 seeds (Jan24 +0.00012, 4/5 folds); public below 49f2, the Jan24 gain did not transfer | not run ‡ | below 0.83187 | `49g_final.csv` |

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
| 2026-10-25 09:45 | 26_ensemble_audience5.csv | 0.821424 |
| 2026-10-25 02:28 | 30_joint_mf.zip (not from this repo's experiments) | 0.818362 |
| 2026-10-25 11:17 | 26_ensemble_hops.csv | 0.822346 |
| 2026-10-25 20:16 | 26_ensemble_combo4.csv | 0.822676 |
| 2026-10-25 21:33 | 26_ensemble_bundle.csv | 0.822709 |
| 2026-10-25 22:42 | 26_ensemble_bundle_2seeds.csv | 0.822773 |
| 2026-10-26 12:34 | 26_ensemble_bundle_4builds.csv | 0.822791 |
| 2026-10-27 15:31 | 34_cosine_s0.csv | 0.823888 |
| 2026-10-27 19:31 | 35_projection_s0.csv | 0.825195 |
| 2026-10-27 22:47 | 35_projection_2seeds.csv | 0.825222 |
| 2026-10-27 22:56 | 35_projection_s1.csv | 0.825160 |
| 2026-10-28 14:23 | 36_combo_s0.csv | 0.825256 |
| 2026-10-28 | 36b_avg_36_35b.csv | 0.825324 |
| 2026-10-29 11:39 | 37c_avg_37x2_36b.csv | 0.825353 |
| 2026-10-29 11:40 | 37b_avg_37_36b.csv | 0.825323 |
| 2026-10-29 11:40 | 37_rank4_2seeds.csv | 0.825336 |
| 2026-10-30 07:15 | 38_aux_act_s0.csv | 0.825251 |
| 2026-10-30 07:16 | 38b_avg_38_37c.csv | 0.825404 |
| 2026-10-30 10:32 | 39_choice_s0.csv | 0.825089 |
| 2026-10-30 10:33 | 39b_avg_39_38b.csv | 0.825545 |
| 2026-10-30 15:15 | 40b_avg_40_39b.csv | 0.825754 |
| 2026-10-30 19:04 | 40_long_mf_s0.csv | 0.825664 |
| 2026-11-02 01:00 | 45s3_aud40_choice04.csv | **0.826069** |
| 2026-11-02 02:10 | 45s6_aud40_choice04.csv | 0.826063 |
| 2026-11-03 07:29 | 47s6_spec_choice.csv | **0.826168** |
| 2026-11-03 14:34 | 48_rank_member.csv | 0.826164 |
| 2026-11-04 19:53 | 49_bayes_stack_s4.csv | 0.829038 |
| 2026-11-04 20:12 | 49_bayes_stack_s8.csv | 0.829240 |
| 2026-11-04 20:35 | 49c_bayes_cat_s8.csv | 0.829782 |
| 2026-11-04 21:32 | 49d_stack.csv | 0.830999 |
| 2026-11-04 22:21 | 49e_view.csv | 0.831691 |
| 2026-11-04 23:16 | 49f_view2.csv | 0.831854 |
| 2026-11-04 | 49f2_tree2.csv | **0.831865** |
| 2026-11-04 | 49g_final.csv | below 49f2 (exact score not recorded) |

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
- **Full-softmax click-log loss** (25 `--aux-full`: each logged click scored against all 50,000
  items, the multinomial likelihood of Mult-VAE/Mult-DAE, instead of 9 popularity-matched
  samples). Clicks-only MF, days 0-5 -> day 6, 3 seeds: 0.74499 (rank corr 0.743 with the MF)
  vs 0.78807 (corr 0.874) with the sampled loss. Without popularity-matched sampling, item
  popularity swamps the taste signal; diverse but uninformative, like 21's deep model. Not run
  in full (about 4.5 h of CPU).
- **Per-user taste correction** (25 `--tune uoff:L`: a learned per-user vector w_u, zero at the start,
  added to the user vector and learned from the impression loss only, with an extra l2 of L x the
  usual one). MF alone, days 0-5 -> day 6, 3 seeds: none 0.81750; L=1 0.80841 (rank corr 0.922 with
  the plain MF); L=10 0.81650 (0.975); L=100 0.81780 (0.993). L=1 has the profile of the sequence
  view (~0.81 alone, corr ~0.92) and learns from a different source (the user's own choices); L=1
  and L=10 are run in full as views.
- **16 seeds for the IDF 1.0 and clicks-only views** (`--seed-offset 8`, averaged with the first 8):
  LightGBM on the current best feature set, half B 0.83842 -> 0.83854.
- **Recency pooling of the event's own history** (25 `--tune recent:k / decay:t`, MF alone, days 0-5 ->
  day 6, 3 seeds): mean 0.81750; last 5 clicks 0.77448 (rank corr 0.835); last 10 0.79818 (0.925);
  weight exp(-position/5) 0.79752 (0.919); exp(-position/20) 0.81535 (0.990). Weaker alone, as
  recency hardly matters inside the fit period; decay 5 has the sequence view's profile and is run
  in full as a view (the sequence features convert to the leaderboard at ~2x their half-B gain,
  possibly through recency at the first event after the cutoff).
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

- **User activity from click-log timestamps does not help on day 6** (`lab/user_activity.py`).
  Motivated by another team's evaluation file (public 0.825): their per-event level tracks the
  user's click rate (clicks per day of history) more than ours, most for users with few train
  events. Features as of each event: known clicks, clicks in 3 / 7 days, history span, clicks per
  day over the last 20, mean gap, warm-up clicks, events so far, train clicks per event so far.
  On the current best feature set (half B 0.83842): all 0.83849, rate subset 0.83850, events
  subset 0.83833. Noise. On the cutoff-5 validation (first events, `lab/first_event_check.py`,
  13 features, 3 LightGBM seeds): 0.81369 -> 0.81378, no segment moves (0 known 0.542 -> 0.538,
  1-5 0.746 -> 0.746, 21+ 0.8376 -> 0.8378). First events do look different (clicks per day over
  the last 20: log -0.15 vs 0.63 in the fit days; mean gap 3.4 vs 2.7 log-hours), but LightGBM
  already handles that through log_hours_since_last / clicks_24h.
- **The other team's systematic difference does not help us.** A regressor learning (their
  score - ours) on evaluation rows from our features explains 16% of the event-level part and 5%
  of the within-event part; adding that learned offset to 13's cutoff-5 validation predictions
  gives 0.81358 (none) / 0.81363 (half) / 0.81346 (full). Whatever earns them +0.005 is in the
  part our features cannot express (diagnostic only; their scores are not used anywhere).
- **Per-user correction views, scored** (`lab/extra_member.py` on the 16-seed views base, half B
  0.83854): + uoff1 0.83868, + uoff10 0.83866, + decay5 0.83845. Small (+0.00014).
  With all 20 sequence caches (12 + 1-layer seeds 8-15) as the base: 0.83848 (no gain from more
  sequence seeds on half B); + uoff1 0.83860; + uoff (uoff1 and uoff10 averaged) 0.83864.
- **Implicit-ALS user vector instead of the mean** (`lab/ials_user.py`, 22's day-6 item vectors,
  dot product alone, mean pooling 0.81278): u = (G + a/n H'H + lam I)^-1 (a/n H'1) with G the
  impression-weighted item Gram. Discounting redundant clicks hurts (a=10 lam=0.01 0.734, a=1
  lam=0.01 0.753); only near-mean settings reach the mean (a=1 lam=10 0.81509, rank corr 0.993;
  lam=100 0.81378, 0.9998). The best is the same light-user shrinkage as 20's alpha 0.9, which
  LightGBM gave back, so not run further. Clicks pointing the same way are real evidence of
  taste strength, not redundancy.
- **The AUDIENCE view: a candidate represented by who clicks it** (26; `lab/audience.py`,
  `lab/audience_feature.py`, `26_audience/audience.py`). Item i's audience A_i = mean over its
  clicks (before day d) of the clicker's taste from their OTHER clicks; B_i = mean taste of the
  users it was shown to. Scores hv . A_i (aud) and hv . (A_i - B_i) (audc), from 22's 1-seed
  vectors of an MF trained on days < d. Day 6, dot alone: MF 0.81278, aud 0.81363, audc 0.81451
  (rank corr 0.945), audc with B weighted 0.5 0.81608, MF + audc 0.81786 (+0.005); rare items
  (1-9 clicks) 0.7635 -> 0.7684-0.7709. Two-hop user side (user = mean audience of their clicks)
  0.813-0.816. As LightGBM features on the current best feature set: **half B 0.83854 -> 0.83973
  (+0.00119), half A 0.83799 -> 0.83910 (+0.00111)**; audc_dot ranks 4th by gain. The first
  clear win since the IDF view: a user-based (who-clicks-it) path that no MF variant has, and it
  helps where the MF is weakest. More vector seeds (26 `vectors.py --vec-seed N`) are next.
  Full ensemble (`lab/ensemble_v5.py`, the 25_ensemble_impronly views with 16-seed IDF 1.0 and
  clicks-only, + aud + audc): LightGBM 0.83973, XGBoost 0.83974, MLP 0.83982, rank average
  **half B 0.84014 (+0.00131 vs 0.83883), half A 0.83939 (+0.00110)**. Built as
  `26_ensemble_audience.csv` (`24_ensemble/ensemble.py --members lgb_full,xgb_full,mlp_full
  --extra ... --extra aud=data/cache/26_aud_c7.npz --extra audc=data/cache/26_audc_c7.npz`).
  5 vector seeds (22's seed 0 + 26 `vectors.py --vec-seed 1..4`, scores averaged; day-6 audc alone
  0.81451 -> 0.81602): LightGBM 0.83993, XGBoost 0.83979, MLP 0.83986, rank average **half B
  0.84025 (+0.00011 over 1 seed), half A 0.83959 (+0.00020)**. Built as
  `26_ensemble_audience5.csv` (caches `data/cache/26s5_{aud,audc}_c7.npz`): **public 0.82142**
  (+0.00109 over 0.82033, 0.77x the half-B gain of +0.00142).
- **More hops of the audience view: the user side helps** (26 `audience.py --variants --hops`, 5
  vector seeds; `lab/extra_member.py` on the current best set incl. aud + audc, LightGBM half B
  0.83993). With C = A - B (audience minus exposure) and u2 = mean of C over the user's known
  clicks: day-6 alone audc05 (A - 0.5B) 0.81764, wb (W - B) 0.81857, hop2 (u2 . W) 0.81923,
  hop2c (u2 . C) 0.81380, hop3 (hv . C2, C2 = the audience operator applied to C) 0.81485,
  hop4 (u2 . C2) 0.81167. As features: audc05 +0.00006, wb +0.00004, hop2 +0.00028, **hop2c
  +0.00057** (0.84050); on top of hop2c: hop2 +0.00005, audc05 -0.00014, **hop3 +0.00030**
  (0.84080), hop4 +0.00010. Scoring well alone did not predict usefulness (hop2c is the weakest
  alone and the best feature). Full ensemble (`lab/ensemble_v5.py`, + hop2c + hop3): LightGBM
  0.84080, XGBoost 0.84076, MLP 0.84094, rank average **half B 0.84122 (+0.00097 vs 0.84025),
  half A 0.84077 (+0.00118)**. Built as `26_ensemble_hops.csv`: **public 0.82235** (+0.00092,
  0.95x the half-B gain).
- **The fixed audience features are saturated** (`lab/extra_member.py`, base = the
  26_ensemble_hops LightGBM set). 10 vector seeds instead of 5 (`26_vectors_c7_s5..9`; day-6 alone
  +0.0000-0.0005): half B 0.84080 -> 0.84082. Audience views from the IDF-pooled MF's vectors (26
  `vectors_pool.py --pool idf --pool-gamma 1.0`, 5 seeds; alone hop2 0.81851, rank corr with MF30
  0.87-0.92): idf_audc 0.84073, idf_hop2c 0.84078, idf_hop2 0.84088, idf_hop3 0.84068. From the
  clicks-only MF's vectors (`--impr-weight 0 --aux-weight 1`; propagation lifts it from 0.776 to
  0.812 alone): clk_audc 0.84092, clk_hop2c 0.84080, clk_hop3 0.84085, clk_hop4 0.84067. All noise:
  the audience path is covered once, whichever MF supplies the vectors.
- **A graph MF trained end to end is much worse** (27 `graph_mf.py`, MF alone, days 0-5 -> day 6,
  3 seeds). Item vector v_i + a*A_i - b*B_i (A, B = 26's audience and exposure as sparse products,
  operators as of each training event's day, per-dimension gates from 0), user vector + c*mean(A - B):
  base 0.81750, graph item 0.80353, graph both 0.80721. Control: one-day batches without the graph
  0.81687, so the batching is not the cause. With no gradient through the propagation (1 seed):
  0.80079 vs 0.80076; the gates grow to |a| ~1.3. The vectors co-adapt with their own audience
  and overfit; the audience only helps computed AFTER an independent MF (experiment 26).
- **The remaining loss is mostly event-level chance** (`lab/decompose.py`, LightGBM on the
  26_ensemble_hops feature set, half B 0.84080). Within-event AUC (mean over clicked events)
  0.830; event level ("does the event get a click", summed probabilities) only 0.692. With a
  perfect event level and our within-event order global AUC would be 0.909. By known clicks:
  0 (0.2% of events) 0.539, 1-5 (4%) 0.774, 6-20 (24%) 0.821, 21-50 (54%) 0.847, 51-100 (18%)
  0.858; event-level AUC 0.63-0.70 everywhere. A separate event-level LightGBM
  (`lab/event_model.py`: event-constant features + per-event max/mean of every row feature) reaches
  only 0.694 vs 0.692, and P(event) x the row model's within-event share gives half B 0.84098-0.84102
  (+0.0002). Whether an event gets a click is mostly noise given the inputs; the reachable gains
  are in the within-event order, where the audience features already went.
- **LightGBM settings are saturated on the wide feature set too** (`lab/lgb_settings_wide.py`, 121
  features of 26_ensemble_hops, lr 0.02, half B): current 0.84080; feature_fraction 0.5 / 0.3
  0.84083 / 0.84085; 127 / 31 leaves 0.84074 / 0.84083; min leaf 1000 0.84084; lambda_l2 10
  0.84080; ff 0.5 + 127 leaves + min leaf 500 0.84087. All within +-0.0001.
- **A learned combiner does not beat the rank average** (`lab/stack.py` on the members of
  26_ensemble_hops, saved by `lab/ensemble_v5.py --save-members`; logistic regression on the
  members' normal-score ranks, fitted on half A): equal weights half B 0.84123, fitted global
  weights (0.23 / 0.48 / 0.79 for LGB / XGB / MLP) 0.84126, weights per known-clicks bucket 0.84121.
  The members agree in every bucket (0-5 known clicks: 0.766 / 0.766 / 0.768, average 0.768).
- **MFs of other sizes add nothing as extra views** (26 `vectors_pool.py --dim 64 / 16 --mf-seeds 8
  --mf-only`; day-6 alone 0.81784 / 0.81619, rank corr with MF30 0.993 / 0.991). On top of the
  26_ensemble_hops set: dim64 half B 0.84093, dim16 0.84079 (base 0.84080). Unlike the IDF /
  clicks-only / impressions-only views they rank like the MF, so they carry no new information.
- **Full-rank item-item audience: weak alone, nothing as features** (26 `item_item.py`: A[i, j] =
  mean over i's clicks of the clicker's other-click share on j, from raw counts; ii1 = mean over
  the user's known clicks of A[i, j], ii2 = the same / popularity of j, ii3 = shrunk log-lift).
  Day 6 alone 0.618 / 0.618 / 0.559 (58% of candidate-history pairs have no overlap; 0.632 on the
  rest; sim_sum alone 0.604), rank corr with MF30 only 0.32. On the 26_ensemble_hops set: half B
  0.84078 / 0.84083 / 0.84087 (base 0.84080). Click information is low-rank; item-level overlap
  adds nothing to the MF-space audience.
- **Repeat candidates on first events** (`lab/repeat_first_events.py`, 13 at cutoff 5): repeats are
  under-predicted on first events (mean prediction / click rate 0.91, vs 0.97 on fit day 4), but
  the best log-odds shift for repeats (+0.2, chosen on the test events, i.e. an upper bound) moves
  AUC 0.81358 -> 0.81362. Repeats are 2.5% of rows.
- **The evaluation set weighs light users ~4x more than day 6.** Known clicks per event,
  evaluation.csv vs fit day 6: 0: 1.3% vs 0.2%; 1-5: 16.4% vs 4.0%; 6-20: 45.0% vs 24.2%; 21-50:
  33.7% vs 54.0%; 51-100: 3.6% vs 17.6%. evaluation.csv has one event per user (the first after
  the cutoff), day 6 has every event, so heavy users dominate it; 2.7% of evaluation users never
  appear in train.csv. Evaluation events by day: 7: 67%, 8: 17%, 9-13: 16%. This explains most of
  the gap between half B (0.841) and public (0.822) and the uneven half-B-to-public conversion.
  `lab/extra_member.py` now also prints an evaluation-weighted half-B AUC (events reweighted to
  the evaluation's known-clicks mix, no evaluation labels involved) and the AUC on 1-20 known clicks.
  **The evaluation-weighted AUC of the current best LightGBM is 0.82374, against public 0.82235**
  for the full ensemble (half B 0.84080): it tracks the leaderboard far better than half B.
- **Rescreen of earlier "noise" views with the evaluation-weighted AUC** (`lab/extra_member.py`,
  base = 26_ensemble_hops set: half B 0.84080, eval-weighted 0.82374, 1-20 known clicks 0.81485).
  Half B / eval-weighted / 1-20: hop2 0.84089 / 0.82402 / 0.81534; hop4 0.84065 / 0.82366 /
  0.81480; wb 0.84080 / 0.82386 / 0.81486; dim64 0.84093 / 0.82402 / 0.81503; idf_hop2 0.84091 /
  **0.82412** / 0.81532; clk_audc 0.84091 / 0.82394 / 0.81505; clk_hop4 0.84075 / 0.82368 /
  0.81472; uoff1 0.84072 / 0.82386 / 0.81489; uoff10 0.84104 / **0.82412** / 0.81513; ii3 0.84087 /
  0.82394 / 0.81501. Views aimed at the user side (hop2, idf_hop2, uoff10) gain slightly more on
  the evaluation mix, but each is at most +0.0004.
  All four best together (hop2 + idf_hop2 + uoff10 + dim64): half B 0.84099, **eval-weighted
  0.82424 (+0.00050)**, 1-20 known clicks 0.81546.
- **Training weighted toward the evaluation mix is worse** (`lab/weight_train.py`: rows weighted by
  (evaluation share / training share) of their known-clicks bucket, to the power p): eval-weighted
  0.82374 (p = 0), 0.82353 (0.5), 0.82321 (1); half B 0.84080 / 0.84064 / 0.84030. Heavy users'
  events teach the model things that hold for light users too.
- **A LightGBM trained on light users only does not help** (`lab/light_model.py`: trained on events
  with <= 20 known clicks, 33% of training rows; its probability averaged into the full model's
  for light rows with weight a): eval-weighted 0.82374 (a = 0), 0.82380 (0.3), 0.82376 (0.5),
  0.82367 (0.7), 0.82342 (1).
- **Full ensemble with the four best evaluation-weighted views** (`lab/ensemble_v5.py`, + hop2,
  idf_hop2, uoff10, dim64; `lab/stack.py` now prints evaluation-weighted AUCs): the uploaded
  26_ensemble_hops members give eval-weighted 0.82432 for the rank average (public 0.82235, half B
  0.84123). With the four views: LightGBM 0.82424, XGBoost 0.82404, MLP 0.82407, rank average
  **0.82466 (+0.00034)**; half B 0.84134 (+0.00012). Not built: below the bar, and with bucket
  weights up to 5.9 the evaluation-weighted AUC is noisier than half B.
- **20 sequence seeds instead of 12, evaluation-weighted** (all `18_mf_c7_*` caches, on the
  26_ensemble_hops set): half B 0.84076 vs 0.84080, eval-weighted 0.82383 vs 0.82374, 1-20 known
  clicks 0.81492 vs 0.81485. Noise on the evaluation mix too.
- **Paired bootstrap intervals** (`lab/paired_ci.py`: half-B events resampled with replacement,
  both models scored on the same resample, 200 resamples). 26_ensemble_combo4 (the four views
  above, full ensemble) vs the uploaded 26_ensemble_hops: plain +0.00012 [-0.00001, +0.00024],
  **eval-weighted +0.00034 [+0.00013, +0.00054], P(better) 1.00**. Paired noise is about +-0.0002:
  a consistent +0.0003 is real, not noise. Built as `submissions/26_ensemble_combo4.csv`
  (Spearman 0.998 with 26_ensemble_hops). `lab/extra_member.py --save-preds` saves every run's
  half-B predictions for these intervals.
- **Earlier MF variants rescreened on the evaluation mix** (on the 26_ensemble_hops set, base
  eval-weighted 0.82374): user vector sum/n^0.9 (20) 0.82394, deep interaction (21) 0.82388,
  choice loss (25) 0.82397, clicks-only IDF 0.82398, clicks-only uniform negatives 0.82397,
  anti-IDF 0.82372. Each +0.0002 at most, single runs.
- **User-activity features rescreened on the evaluation mix** (`lab/activity_eval.py`, features of
  `lab/user_activity.py` on the 26_ensemble_hops set): all 0.82393, events subset 0.82386, rate
  subset 0.82387 vs 0.82374; 1-20 known clicks unchanged (0.81488-0.81493 vs 0.81485).
- **The MF-strength mismatch between fit days and the test is real but harmless**
  (`lab/mf_strength.py`). Fit day d is scored by an MF trained on d days: MF13 alone day 2 0.781 ->
  day 6 0.819 at an unchanged score scale (sd 1.27-1.30; test rows 1.36), so LightGBM learns an
  average trust from weaker MFs than the test gets. Row weights (day - 1)^p: eval-weighted 0.82361
  (p = 1), 0.82324 (p = 2) vs 0.82374; the scoring day as a feature (unseen days act like the latest):
  0.82388, paired interval +0.00014 [-0.00009, +0.00034]. Not kept.
- **Bundling the small views pays** (`lab/extra_member.py --save-preds` + `lab/paired_ci.py`,
  LightGBM). combo4 + choice + clicks_idf + clicks_unif + uvec09 + deep vs combo4: half B 0.84118 vs
  0.84099, +0.00020 [+0.00006, +0.00032]; eval-weighted 0.82445 vs 0.82424, +0.00021 [-0.00003,
  +0.00045]. LightGBM alone has gone 0.82374 -> 0.82445 eval-weighted (+0.0007) since the upload.
  Full ensemble (`lab/ensemble_v5.py`, all 18 extra groups): LightGBM 0.84118, XGBoost 0.84114,
  MLP 0.84128; rank average vs the uploaded 26_ensemble_hops members: **half B 0.84162, +0.00039
  [+0.00025, +0.00054]; eval-weighted 0.82498, +0.00066 [+0.00041, +0.00088]**. Built as
  `submissions/26_ensemble_bundle.csv`.
- **The evaluation-weighted half B predicts the leaderboard.** 26_ensemble_combo4: predicted
  +0.00034 (eval-weighted, vs 26_ensemble_hops), public **+0.00033** (0.822346 -> 0.822676); plain
  half B said +0.00012. Use the evaluation-weighted AUC with its paired interval for decisions.
  26_ensemble_bundle: predicted +0.00066 vs 26_ensemble_hops (+0.00032 over combo4), public +0.00036
  (+0.00003 over combo4). The five extra views' share did not transfer: one hit, one miss, so treat
  the metric as good to about +-0.0003, and the public split itself has noise of that order.
- **Averaging the three latest submissions is worse than the best alone** (hops + combo4 + bundle,
  `15_more_seeds/average.py`; day-6 members' rank averages averaged): vs the bundle, plain -0.00010
  [-0.00018, -0.00003], eval-weighted -0.00019 [-0.00032, -0.00006]. Not uploaded. Seed averaging of
  the SAME ensemble instead: 24 `ensemble.py --model-seed N` (LightGBM / XGBoost seed, MLP seeds from
  1000 N; 0 reproduces the earlier files) -> `26_ensemble_bundle_seed1.csv`, averaged with seed 0 as
  `26_ensemble_bundle_2seeds.csv`.
- **Broad MF search** (27 `graph_mf.py --tune`, graph off = 13's MF, days 0-5 -> day 6, 1 seed; base
  0.81370). Bigger and more regularised is worse: dim 64 + l2 0.1 0.80712 (5000 steps 0.80361), dim
  128 + l2 0.2 / 0.3 0.75419 / 0.59962, dim 96 + l2 0.15 + lr 0.002 0.79543, dim 64 + batch 2000 + l2
  0.1 0.80680, dim 64 + aux 0.2 + l2 0.1 0.80750, dim 16 + l2 0.03 0.80920, batch 2000 0.81320.
  **Slower learning for longer is better: lr 0.0015 + 6000 steps 0.81656 (+0.0029)**, dim 64 with
  the same 0.81579. Earlier tuning changed lr only at 3000 steps (lr 0.002: 0.8141). 3 seeds: base
  0.81750; lr 0.0015 + 6000 steps 0.81857 (+0.0011); **lr 0.001 + 9000 steps 0.81869 (+0.0012)**;
  lr 0.0015 + 8000 0.81759; lr 0.002 + 5000 0.81819. **But seed averaging already provides it**:
  MF30 (30 seeds, current schedule) scores 0.81921 alone on day 6, the slow schedule with 5 seeds
  (26 `vectors_pool.py --lr 0.0015 --steps 6000 --mf-seeds 5 --mf-only`) 0.81896, their average
  0.81928; rank corr 0.997. As a view on the 26_ensemble_bundle set: +0.00001 [-0.00020, +0.00021]
  eval-weighted. The core MF is at its optimum for this model form.
- **Out-of-fold MF features are WORSE than time-forward ones** (28 `oof_mf.py --cutoff 6` +
  `lab/oof_check.py`; 5 folds by event over days 0-5, each scored by an MF trained on the other folds
  with their label clicks removed from the click log; 3 seeds; day 6 scored by the MF on days 0-5 in
  all variants). Out-of-fold MF alone: 0.814-0.818 on every day, vs time-forward 0.776 (day 2) to
  0.812 (day 5). LightGBM on base + MF features, half B / eval-weighted: time-forward days 2-5
  0.83441 / 0.81785; out-of-fold days 2-5 0.83392 / 0.81723; out-of-fold days 0-5 0.83391 / 0.81727;
  paired interval (0-5 vs current) plain -0.00050 [-0.00067, -0.00029], eval-weighted -0.00058
  [-0.00086, -0.00020]. An out-of-fold MF has seen the user's later behaviour (later histories carry
  the held-out clicks as inputs), so training rows' scores relate to labels differently from the
  test rows', which have no future. The time-forward rule in AGENTS.md is right; days 0-1 add nothing.
- **Leaderboard on 2026-10-25: 27th of 27 at 0.822346; first place 0.832734**, 25 teams between
  0.8227 and 0.8320. A shared gap of up to 0.010 means a missing signal, not tuning. Checked and
  NOT the cause:
  - No cold items: every evaluation candidate was shown in train.csv, 0.02% never clicked, 1.0%
    with <= 3 clicks (day 6: 1.2%). The catalogue is static (no item is first shown after day 0).
  - Event-level calibration on first events (`lab/first_event_calibration.py`, 13 at cutoff 5,
    diagnostic): predicted / actual event click share is flat across known clicks (0.76-0.79), hours
    since the last click (0.76-0.80), new vs returning users (0.75 / 0.77) and event day (0.75 /
    0.77). No segment is mis-ranked against another.
  - No context effects: share of events with a click by time bucket 0.627-0.632, weekend 0.629 vs
    weekday 0.629; item click rate per time bucket predicts day 6 worse than overall (0.520 vs
    0.528), weekday-only or weekend-only item rates no better. Day 6 is a Sunday and the evaluation
    starts on a Monday, but nothing depends on it.

- **Raw-data pass: see DATASET.md** (2026-10-26; `lab/dataset_*.py`). New: histories are exact
  between consecutive events (99.73%), so users have no unobserved events; per-user click
  propensity is 71% real variance; candidates are re-shown to a user 8x more than chance but
  re-shown items click at their usual rate; the raw fall of click share with time since the last
  click (0.654 -> 0.571) is user confounding, and the within-user effect matches a permutation null
  (hot-hand bias): no recency.

- **Output-level blending and stacking add nothing here** (`lab/output_blend.py`, day-6 members of
  26_ensemble_bundle + each standalone model's own time-forward score, calibrated to a logit on half
  A). Blending one standalone model into the members' mean logit: the weight chosen on half A is 0
  for MF30, sequence, choice, IDF, impressions-only, audience, hop2c, uoff10, deep and 0.02 for
  clicks-only (+0.00000). All ten with logistic weights: eval-weighted +0.00012 [-0.00005,
  +0.00028]. LightGBM stacker (members + standalone logits + event-relative + known clicks, user
  rate, events; trained on half A): -0.00024 [-0.00045, -0.00007], with an eval-weighted loss
  -0.00056. Every standalone model is already a feature of the members, so its information has
  been used; blends help when separately built pipelines are combined, which our single feature
  set already does inside LightGBM.

- **A true SASRec (next-click objective) is weak here** (29 `sasrec.py --quick`: causal transformer,
  dim 64, 2 layers, [START] token, windows of 50 clicks, sampled softmax with 512 popularity
  negatives and log-Q correction; trained on the clicks before day 6, 15 epochs). Experiment 18 was a
  transformer USER ENCODER with the MF's losses, not a next-click model. Day 6 alone 0.67227
  (within-event 0.678; rank corr with MF30 0.47); blended into MF30's score it hurts at every
  weight (0.81921 -> 0.81887 at 5%, 0.81187 at 30%). 60 epochs on the Apple GPU (`--epochs 60
  --device mps`, loss 4.17 -> 3.23): alone 0.69617, blended 0.81918 / 0.81873 / 0.81687 at 5 / 10 /
  20%. Not undertraining: the next-click objective fits this simulator badly. Each click is learned
  only from the clicks BEFORE it, while taste here is the unordered set of clicks (DATASET.md), so
  the best context is all OTHER clicks, which is exactly experiment 10's change (+0.008 public).
  SASRec / Mamba4Rec-style models win where order and recency matter; here they do not.

- **Splitting our own feature set into separate pipelines does not help** (`lab/split_pipelines.py`,
  LightGBM each on the 40 core features + MF30 + one family, 26_ensemble_bundle set). Half B /
  eval-weighted: full 0.84118 / 0.82445; MF family 0.83881 / 0.82168; audience family 0.84013 /
  0.82333; sequence 0.83697 / 0.82018. Rank corr with full 0.984-0.997 (they share the core and
  MF30). Blends vs full, eval-weighted: equal weights -0.00025 to -0.00141; weights fitted on half A
  give full ~all the weight, +0.00000 to +0.00002 (full + all three -0.00003). The audience family
  alone is almost the full model.

- **Learned ID embeddings in the MLP add nothing** (`lab/nn_ids.py`: the MLP member's inputs + a
  candidate item embedding (dim 16) + item bias + a separate embedding table averaged over the user's
  known clicks + their product, trained on days 2-5). Overfits after one epoch (half A 0.838 ->
  0.835 -> 0.828); 2 seeds alone half B 0.83877 / eval-weighted 0.82136 (MLP member 0.84128),
  rank corr 0.988 with the bundle; blend weight chosen on half A 0, +0.1 -0.00002, +0.3 -0.00030.

- **Mult-VAE on click sets is weak** (30 `multvae.py --quick`: binary click set -> 600 -> latent 200
  with a Gaussian prior -> softmax over all items, beta 0.2, 60 epochs on the clicks before day 6).
  Day 6 alone 0.70532 (MF30 0.81921), rank corr 0.54; worse in every group, including 1-5 known
  clicks (0.665 vs 0.753); blended into MF30 it hurts at every weight (0.8188 at 0.1).
- **Click-only objectives are the weak ones here.** Day 6 alone: MF30 (impressions + clicks)
  0.819; impressions-only MF 0.786; clicks-only MF 0.794; Mult-VAE 0.705; SASRec 0.696. Which of the
  10 shown candidates gets clicked is learned from the impressions; models of "what this user clicks
  in general" miss it. Industry sequence/VAE models do not transfer to this impression-labelled task.

- **The user's impression history is no taste estimate** (`lab/shown_profile.py`, day 6, MF vectors
  of days 0-5: candidate . (mean vector of every item shown to the user before - the global mean)).
  Alone 0.59267 (MF dot 0.81278), blended into the MF dot it hurts at every weight (0.80916 at 0.1).
  Light-click users are light-impression users too (median earlier impressions: 1-5 known clicks
  20, 21-100 250); for heavy users the shown and click profiles agree only at cosine 0.42.

- **A variational user encoder in the impression MF does not beat the plain mean** (31 `vae_mf.py`,
  user vector = mean of clicked items' vectors + a sampled correction from a click-set encoder with
  a KL prior, 3 seeds, days 0-5 -> day 6; base 0.81750). With the standard N(0, 1) prior every variant
  collapsed to ~0.56 (noise ~30x the vectors' scale; the model fell back to its biases). With a prior
  on the MF's scale: s 0.05 (beta 0.1 or 1) 0.81596, s 0.2 0.76978; no prior (control) 0.80835. The
  tighter the prior, the closer to the plain MF; no setting beats it, light users included (1-5
  known clicks 0.7504 vs 0.7525). Taste behaves like the plain mean of the clicked items' vectors.

- **Event-level ceiling and within-event centrality** (2026-10-26). With the TRUE per-user click
  propensity alone (Beta, mean 0.63, sd 0.138 as measured in DATASET.md), event-level AUC would be
  0.668; ours is 0.692, so the event level is at its ceiling. A candidate's similarity to the other 9
  of its own event (`lab/set_centrality.py`, MF vectors): 0.500 overall, 0.502 within events, chance
  in every known-clicks group; hurts the MF dot at any weight.

- **Where item-vector error costs** (MF30 on day 6, `logs/2026-10-26_lab_item_error_diag.txt`). By the
  popularity of the user's clicked items (terciles): 1-5 known clicks 0.7431 / 0.7533 / 0.7641, 6-20
  0.7910 / 0.8052 / 0.8088, 21-100 0.8244 / 0.8319 / 0.8338: light users lose mainly through having
  few clicks, not through their items' vectors. By candidate clicks: 0-3 (1.2% of rows) 0.683, 4-10
  (12.7%) 0.778, 11-30 (47.7%) 0.819, 31+ (33.9%) 0.842: rare candidates are the item-side weak spot.

- **45 core MF seeds instead of 30** (seeds 30-44 by 26 `vectors_pool.py --seed-offset 30 --mf-seeds 15
  --mf-only` = 13's MF; `lab/extra_member.py --mf-core`, bundle set): half B 0.84118 vs 0.84118,
  eval-weighted 0.82455 vs 0.82445, +0.00010 [-0.00007, +0.00028]. Seed averaging is saturated.
  `24_ensemble/ensemble.py --mf-core-extra` builds with it.
- **Rare candidates are the final model's item-side weak spot** (bundle-set LightGBM, half B, by the
  candidate's clicks before the day): 0-3 (1.0% of evaluation rows) AUC 0.727; 4-10 (15.2%) 0.803;
  11-30 (47.3%) 0.839; 31+ (36.4%) 0.859. Calibrated (mean prediction = click rate in every group);
  a clicked rare item ranks 2.06th of 10 on average vs 1.26th for popular ones.

- **Per-item Bayesian refit of the item vectors does not help** (`lab/item_refit.py`: each item's
  vector re-fitted from its own impressions before day 6 by penalised logistic regression on the
  users' taste vectors, prior mean = the MF's vector, strength lam). Day 6: MF dot 0.81278 (rare
  candidates 0.7688); refit dot lam 3 / 30 / 300: 0.81090 / 0.81321 / 0.81285 (rare 0.7666 / 0.7688
  / 0.7688); with a refitted bias 0.807; every blend into MF30 (0.81921) is lower (0.8170-0.8174).
  The MF's item vectors already use each item's data fully.

- **The wider test score spread is the user mix, not MF strength** (`logs/2026-10-26_lab_test_scale.txt`).
  MF30 logit sd: day 6 1.266, day 6 reweighted to the evaluation's known-clicks mix 1.350, test
  1.362; per known-clicks group day 6 and test agree to 0.01-0.03. No train/test scale shift to fix.

- **First-of-day events need no special treatment** (`lab/first_of_day.py`; a user's first event of a
  day is the fit-period analogue of an evaluation event, 36% of rows). Bundle-set LightGBM, half B /
  eval-weighted / first-of-day only (eval-weighted): current 0.84118 / 0.82445 / 0.81278; + position
  in the day and hours since the previous event 0.84109 / 0.82433 / 0.81253; first-of-day rows x2
  0.84107 / 0.82433 / 0.81257; x4 0.84081 / 0.82423 / 0.81242.

- **The impression loss does not need more user context** (32 `allctx_mf.py`: training events see the
  user through ALL their clicks before the scored day minus the event's own click, instead of the
  clicks before the event; scoring unchanged; 3 seeds, days 0-5 -> day 6): base 0.81750, all-clicks
  context 0.81768; light users unchanged (1-5 known clicks 0.7525 / 0.7525).

- **Ranking objectives over MANY events at once** (`lab/global_rank.py`, bundle set; groups = random sets
  of G events, so the objective sees cross-event pairs, a surrogate of global AUC; one event per group
  was worse earlier). Half B / eval-weighted: binary 0.84118 / 0.82445; rank_xendcg G50 0.84101 /
  0.82420, G500 0.84087 / 0.82416; lambdarank G50 0.84122 / 0.82441, **G500 0.84137 / 0.82467**.
  Larger groups and intervals: `lab/global_rank2.py`.

- **Multi-interest user vectors are worse** (`lab/multi_interest.py`, day 6, MF vectors of days 0-5: users
  with >= 6 known clicks split into k = 2 clusters by k-means, candidate score max(u1 . v, u2 . v)).
  Users with >= 6 clicks: single mean 0.81640, k=2 max 0.78994, halfway mean-max 0.80826. Taste is one
  direction (as the attention test found).

- **Ideas aimed at light users and rare candidates** (`lab/light_rare_ideas.py`, day 6, vs MF30 0.81921;
  groups: 1-5 known clicks 0.7532, candidates with <= 10 clicks 0.7726). Raw item-item co-clicks (ii3)
  blended at 0.1: 0.82026 (+0.0011; 1-5 known clicks +0.0015, rare candidates +0.0011), but as a
  LightGBM feature it added nothing (eval-weighted 0.82394 vs 0.82374) -- the co-click features carry
  it. History expansion scaled by sparsity hv + lam/(n+k) u2: 0.8117-0.8126 vs 0.81278 (1-seed dot).
  Item click rate among light users only: 0.517 alone (all users 0.528), hurts blends. Rare items'
  vectors shrunk toward k-means (200) cluster centres: 0.8066-0.8129 vs 0.81278. None kept.

- **Cold start is a small prize; ordinary users decide the score** (`lab/segment_headroom.py`: the best
  model's half-B scores ranked PERFECTLY within one segment, the rest untouched; evaluation-weighted
  AUC 0.82498). Gain if perfect / if 30% of the way: users with 0 known clicks (1.3% of evaluation
  events) +0.0006 / +0.0002; new users without earlier train events (2.7%) +0.0036 / +0.0012; 1-5
  known clicks (16.4%) +0.022 / +0.007; 6-20 (45.0%) +0.058 / +0.019; 21+ (37.4%) +0.047 / +0.015;
  candidates with <= 10 clicks +0.051 / +0.026 (inflated: also pushes every non-clicked rare
  candidate to the bottom). Cold-start fixes (first-click popularity, propensity prior) cannot reach
  the +0.0034 to 15th place; the prize is ranking quality for users with 6-50 clicks.

- **A dedicated fallback model for light users adds nothing** (`lab/fallback_model.py`: LightGBM on
  history-dependent features recomputed from up to 5 known clicks -- MF taste match and cosine,
  two-hop audience matches, repeat flag, popularity taste, gap to the event's best -- plus item and
  user features; trained on ~390k SYNTHETIC light users (every day 2-5 event replayed with only its k
  most recent clicks, k = 0-5) and the ~21k real ones). Real day-6 half B: 0 known clicks (108 events)
  best model 0.5360 vs fallback 0.5297; 1-5 known clicks (1,974 events) 0.7774 vs 0.7717. Blended
  into the best model for those users: +0.00005 (25%), -0.00005 (50%), -0.00066 (100%). The main
  model already learns light users as well as a specialised one.
- **Rebuilding the best submission:** `experiments/26_audience/build_best.sh` holds the exact commands and
  caches behind 26_ensemble_bundle_4builds.csv, and checks the rebuilt file's rank correlation with
  the uploaded one (>= 0.995).

- **A time-forward stack adds nothing** (`lab/tf_stack.py`: level 1 = LightGBM on the bundle set trained
  on days 2..d-1 predicting day d, for d = 3-6 (AUC 0.822 / 0.828 / 0.837 / 0.841); level 2 = small
  LightGBM on days 3-5 with the level-1 logit, its event rank and gap, and 10 raw features). Day-6
  half B: level 1 0.84093 / eval-weighted 0.82423, level 2 0.84095 / 0.82423. Stacking our model on
  itself cannot explain a 0.824-on-day-2 pipeline; its gain was not the stacking alone.

- **No raw signal explains the best model's errors** (`lab/signal_search.py`: 13 per-row signals the model
  does not see directly -- user earlier events, distinct clicked items, repeat-click share, warm-up
  share, hours since first click / previous event, previous event clicked; candidate shown before /
  in the previous event; item clicks in the last 24 h, age, repeat-click share, light-user share --
  offered to a LightGBM that starts from the best model's logit, fitted on half of half B's events).
  Held-out: 0.84229 / eval-weighted 0.82574 -> 0.84229 / 0.82575 (1 tree); every signal's correlation
  with the residual within +-0.008. **Pitfall:** a first version computed the warm-up share over ALL
  the user's clicks, including later ones and the event's own click: +0.0064 eval-weighted, all of
  it leakage (it would do nothing on the leaderboard, whose histories hold no post-cutoff clicks).
  Every per-user statistic must be as of the event.

- **Research: the 2026 Arena is framed as sequential recommendation "adapting to changing user behavior",
  and published simulators often move taste toward each clicked item (arXiv 2403.07571). Tested on
  the data, taste does NOT drift** (`lab/mild_decay.py`, day 6, MF vectors of days 0-5, user vector =
  exp(-age/tau)-weighted mean of the known clicks): tau 10 / 30 / 50 / 100 / 200 / 1000 clicks:
  0.80641 / 0.81197 / 0.81252 / 0.81275 / 0.81279 / 0.81279 vs plain mean 0.81278; decay by time
  (24-500 h) 0.761-0.812. Earlier tests covered only strong recency; mild decay converges to the
  plain mean and never beats it.
- **Taste is a SET: repeated clicks should count less** (`lab/taste_set.py`, same setting; 89% of
  histories repeat an item): multiset mean (current) 0.81278, unique items 0.81304, unique items
  weighted by sqrt(times clicked) **0.81330** (+0.0005, every known-clicks group but 1-5), no 100 cap
  0.81282. **MF trained with that pooling** (33 `set_pool.py --tune`, 3 seeds, days 0-5 -> day 6): count
  (= 13) 0.81750, sqrt 0.81804 (+0.0005, every known-clicks group), unique 0.81753. **As a view**
  (33 `sqrt_view.py`: sqrt-set-pooled taste match from the 10 saved vector seeds of 22 / 26, plus its
  difference to the count-pooled one; day 6 alone 0.81812, rank corr with MF30 0.951) on the bundle
  set: half B 0.84118 -> 0.84157, **+0.00038 [+0.00023, +0.00054]**; eval-weighted 0.82445 -> 0.82477,
  **+0.00032 [+0.00002, +0.00061]**. Repeated clicks over-count in the mean; the repeat-count features
  did not recover this. Exponent sweep (`lab/set_alpha.py`: unique item weighted by count^alpha, 5 vector
  seeds, day 6): alpha 1 (current) 0.81717, 0.75 0.81754, **0.5 0.81765**, 0.25 0.81758, 0 0.81738,
  -0.25 0.81711. Trained as a full view: 26 `vectors_pool.py --pool sqrtset --mf-seeds 8 --mf-only`.

## Ideas not tried yet
- Empty-history events (~1% of rows) score ~0.55: a popularity-only fallback might help.

- **The full sqrt-set MF adds nothing; the quick sqrt view does** (2026-10-27, `logs/2026-10-27_lab_sqrtmf_view.txt`,
  bundle set via `lab/bundle_args.sh`). 26 `vectors_pool.py --pool sqrtset --mf-seeds 8 --mf-only`
  (`26p_mf_c7_20878df28896.npz`): day 6 alone 0.81913 but rank corr 0.9953 with MF30, so like dim64 it ranks
  as the MF: half B 0.84112 / eval-weighted 0.82429 (base 0.84118 / 0.82445). The quick view
  (`33_sqrtpool_c7.npz`, rank corr 0.951) reproduces +0.00038 / +0.00032: keep it for the next build.
- **Taste does not depend on context** (`lab/context_taste.py`, day 6, MF vectors of days 0-5; m clicks drawn from
  the event's own time bucket / day type and m from the others). Time bucket m=3 0.75369 vs 0.75313, m=6
  0.78401 vs 0.78506; day type m=3 0.75093 vs 0.75508, m=6 0.78603 vs 0.78674. Weighting clicks by hour-of-day
  proximity only hurts (kernel 6 h 0.81050, 1.5 h 0.78136 vs 0.81310 unweighted).
- **Warm-up click intensity as a propensity estimate: real but already covered** (`lab/warmup_propensity.py`,
  `lab/warmup_feature.py`). Event rates are stable from warm-up to train (log corr 0.77), and warm-up clicks per
  day / train events per day correlates 0.545 with the train click rate of users with >= 20 events (half of
  their own events: 0.586). But for the users who need it, the event rate is as noisy as the click rate:
  event-level AUC on half B, pooled estimate vs user_rate, 0-5 train events 0.576 vs 0.583. As six features on
  the bundle set: half B 0.84124 / eval-weighted 0.82446, paired +0.00002 [-0.00021, +0.00028].
- **PureSVD is popularity-dominated** (`lab/pure_svd.py`; truncated SVD of the sqrt click matrix before day 6,
  columns weighted (pop + c)^(-alpha/2), users folded in from their known clicks). Unsmoothed alpha 1 is
  unstable (rank 40 0.592, rank 64 0.791, alpha >= 1.5 at chance); smoothed (c 5-50) 0.68-0.75 alone, rank corr
  0.47-0.74, no blend gain over MF30 (0.81959 best vs 0.81921). A spectral view of clicks alone has the same
  failure as EASE: without impression-matched negatives, popularity swamps taste.
- **The evaluation candidate policy is the training policy** (`lab/eval_policy.py`, per known-clicks group,
  day 6 / day-6 first events / evaluation): repeats 2.4-2.8% in all three, re-shown 0.2% / 0.4-0.5% / 1.4-2.3%
  (heavy users: evaluation 1.5% between day 6's 2.3% and first events' 1.4%), cos(candidate, taste)
  0.035-0.040 everywhere (random items 0.000).
- **No taste shift at the warm-up -> train boundary** (`lab/boundary_shift.py`, day 6, m clicks from each period):
  warm-up vs train-period clicks m=3 0.74897 vs 0.75779, m=6 0.78466 vs 0.78904, m=10 0.80281 vs 0.80482 --
  the small gap is expected from fitting the item vectors on the train-period impressions. Age control (oldest
  vs newest m train-period clicks) goes the OTHER way: 0.76117 vs 0.74946 (m=3), 0.78898 vs 0.78443 (m=6),
  consistent with newer clicks being repeats more often. Nothing suggests a shift at the evaluation boundary.
- **Taste does not predict click propensity** (`lab/taste_propensity.py`, 7,824 users with >= 20 events before day 6,
  click-rate split-half reliability 0.581). 5-fold ridge on features of their known clicks (MF vectors of days
  0-5): mean vector (32 dims) R^2 -0.005, its norm +0.017, coherence (norm of the mean unit vector) +0.034, mean
  log popularity -0.000, log known clicks +0.247 (mostly the click rate itself). No route from a light user's few
  clicks to their propensity.
- **Item vectors learned as light users see them** (26 `vectors_pool.py --user-wpow p`: each training event's
  impression loss weighted by 1/(user's training events)^p, mean 1; MF alone, days 0-5 -> day 6, 3 seeds):
  base 0.81750; p 0.5 0.81299 (rank corr 0.989); p 1.0 0.80383 (0.959). p 1.0 has the profile of the useful
  views (IDF 1.0: 0.815 / 0.958) and is run in full (`--user-wpow 1.0 --mf-seeds 8 --mf-only`).
  Full run (`26p_mf_c7_6a2b62de353d.npz`): day 6 alone 0.80577, rank corr 0.962 with MF30, 0.924 with the sequence
  model. On the bundle set: half B 0.84118 -> 0.84129, eval-weighted 0.82445 -> 0.82443, 1-20 known clicks 0.81573
  -> 0.81566; best feature ranks 39th by gain. Diverse but not informative: weighting users equally in the item
  vectors adds nothing, light users included.

- **A replica of the simulator: large theoretical headroom, but the item side is information-limited**
  (2026-10-27; `lab/replica_sim.py`, `lab/replica_bench.py`, `lab/replica_item_limit.py`, logs
  `2026-10-27_lab_replica_*`). Truth built from our own model: item vectors / biases of the MF on days 0-6,
  taste = unit sqrt-set mean of the user's real clicks' vectors x R x scale, propensity drawn from each user's
  Beta posterior, choice over the 10 candidates + no click; every training event's click regenerated on the
  real timeline (real candidates, timestamps, warm-up clicks). Row CTR 0.0617-0.0643 and click share 0.617-0.643
  match the real 0.0629 / 0.629 at every scale. **Calibration**: our MF (13, 3 seeds, days 0-5 -> day 6) scores
  0.748 / 0.818 / 0.831 / 0.878 / 0.903 at scale 1.0 / 1.42 / 1.5 / 2.0 / 2.5; the real data's 0.8175 matches
  scale 1.42 (0.8178). There, day 6: oracle 0.886 (event level 0.738, within event 0.877); true taste logit
  alone 0.873; true item vectors with the user = pooled true vectors of their known clicks 0.859; MF 0.818.
  So in a world shaped like our model and as hard for our MF as the real data, **~0.04 is lost to item-vector
  estimation** (MF scores correlate 0.824 with the true taste logit: 0.49 for candidates with 0-3 clicks,
  0.71 for 4-10, 0.82 for 11-30, 0.88 for 31+).
  - The audience correction W + gamma (A - beta B) recovers part of it, as on the real data: 0.81302 ->
    0.81746 (gamma 2, beta 1; corr with the truth 0.824 -> 0.835), i.e. ~10% of the item-side gap.
  - **But the gap is mostly not recoverable from the data.** Re-fitting every item from its own impressions of
    days 0-5 by penalised logistic regression with the TRUE user tastes and every other term of the true logit
    as an offset (the information limit of the impressions): 0.80653 (corr 0.766; 0-3 clicks 0.30, 4-10 0.56),
    with the prior-matched penalty 0.78396 -- WORSE than one MF seed (0.80894, corr 0.811). The MF already
    gets more out of the data than an oracle-user regression on the impressions, through the history role and
    the click log. From the MF's own user vectors (one alternating step): 0.78179.
  - Conclusion: the replica's oracle is far above what any estimator can reach; ~140 impressions and ~8 clicks
    per item cannot pin down a 32-dim vector. A +0.01 gain would need structure in the TRUE item space that a
    Gaussian 32-dim prior misses (fewer effective dimensions, clusters, sparsity) -- the replica, built from our
    own vectors, cannot show that by construction. Caveat: the replica's tastes are built from the users' real
    clicks, so its warm-up clicks agree with its tastes by construction, which flatters the MF slightly.
- **Posterior predictive checks against the replica** (`lab/replica_ppc.py`, logs `2026-10-27_lab_replica_ppc*.txt`).
  The replica holds only our mechanisms, so a statistic that differs between real data and replica points at a
  missing one. Fair version: each world's own MF vectors of days 0-5, statistics on day-6 clicks vs the user's
  click k steps earlier. Cosine lag 1/2/3/5/10/20: real 0.195 / 0.201 / 0.204 / 0.205 / 0.198 / 0.192, replica
  0.184 / 0.191 / 0.195 / 0.195 / 0.193 / 0.186 -- the same shape (a uniform +0.01 level shift), so **no
  sequential mechanism**. Exact re-click of the item k steps back: real 2.14% / 1.84% / 1.31% (k = 1 / 2 / 5),
  replica (no repeat bonus) 1.36% / 1.26% / 1.01%: real repeats favour the most recent clicks (excess 1.57x /
  1.46x / 1.30x). LightGBM already has rep_recency, so this is likely covered. (A first version measured the
  real side with vectors fitted on those very clicks: in-sample coherence 0.29 vs 0.22, not comparable.)
- **Repeat position in clicks adds nothing** (`lab/repeat_position.py`, `lab/repeat_feature.py`). On half B the bundle
  LightGBM is near-calibrated by how many clicks back a repeat candidate was last clicked (click rate / prediction:
  1 back 1.068, 2 back 1.040, 3-5 1.033, 6-20 0.973, 21-100 0.956; 2.7% of rows; AUC within repeats 0.7134). As
  features (log position, is-last, occurrences among 100 known clicks) on bundle + sqrtview: half B 0.84157 ->
  0.84152, eval-weighted 0.82477 -> 0.82472, paired -0.00004 [-0.00028, +0.00019].
- **Splitting and stacking, for the record** (asked again 2026-10-27): already tested three ways -- split pipelines
  (`lab/split_pipelines.py`), output-level stacking (`lab/output_blend.py`), time-forward stack (`lab/tf_stack.py`) --
  all within +-0.0002. The single LightGBM already sees every model as features, so it stacks internally on more rows
  than a stacker gets.
- **Adversarial validation finds one train/test inconsistency, and it is harmless** (`lab/adversarial.py`,
  `lab/stale_fix.py`). A classifier separating evaluation rows from each user's first day-6 event (reweighted to the
  evaluation's known-clicks mix) on 13's features: apart from day_type (day 6 is a Sunday) and the legitimate gap /
  history-length differences, item_log_pop_decay separates (AUC 0.359; median 0.39 on evaluation vs 0.67 on day 6):
  it is a 24 h-decayed click count refreshed from every event's history during the fit period, but after the cutoff
  it only decays (a day-13 event sees e^-6 of it). rep_recency uses the same 24 h clock. Fixed exactly post hoc
  (values frozen at the cutoff; nothing updates after it) and checked on the cutoff-5 validation, whose test events
  are 0-2 days stale: original 0.81358; pop frozen 0.81357 (-0.00001 [-0.00006, +0.00006]); rep_recency frozen
  0.81359; both 0.81359 (+0.00001 [-0.00005, +0.00008]); day-6 test events 0.79727 -> 0.79730. LightGBM does not
  lean on either feature in a way that costs anything. All other features match between the sets (MF scores,
  CTRs, user rate, relative features: per-feature AUC 0.48-0.52).
- **The tied MF is not distorted by repeats** (26 `vectors_pool.py --tune rep / repx`, 3 seeds, days 0-5 -> day 6). With
  one vector per item and no repeat term, the MF can only explain re-clicks (29% CTR, ~12% of clicks) through the
  candidate's own vector inside the pooled history (|v|^2/n), and the click-log context keeps the target's other
  occurrences. rep = explicit repeat bias (1 / 2 / 3+ occurrences) + candidate excluded from its own pooled history;
  repx = also the target's occurrences dropped from the click-log context. All / non-repeat rows / repeat rows: base
  0.81750 / 0.80767 / 0.66980; rep 0.81763 / 0.80761 / 0.67545 (rank corr 0.999); repx 0.81728 / 0.80721 / 0.67500.
  Taste geometry unchanged; the MF already separates the repeat effect well enough.
- **No log-Q correction in the click-log softmax is right** (26 `vectors_pool.py --tune logq:a`: each candidate's logit
  minus a x log of its sampling probability, as standard for sampled softmax; 3 seeds, days 0-5 -> day 6). All /
  non-repeat rows: base 0.81750 / 0.80767; logq 0.5 0.81341 / 0.80319; logq 1 0.80655 / 0.79576 (rank corr 0.965);
  logq 1 + aux 0.2 0.79992. The task is P(click | shown), and candidates are drawn from the impression distribution,
  so impression-weighted negatives without correction mimic the candidate sets; the correction targets a
  "click among all items" objective instead (like the full softmax, 0.745 for clicks-only).
- **The dot product passes vector-norm noise into every score: COSINE is +0.005 alone, and a new view** (2026-10-27).
  `lab/score_form.py` (day 6, one seed of 22's vectors of days 0-5): dot 0.81278, item-normalised 0.81347,
  user-normalised 0.81677, cosine **0.81777** (within-event 0.8201 vs 0.8191; every known-clicks group), unit-vector
  mean 0.81420, -distance^2 0.74617. The gain holds at every seed count (`logs/2026-10-27_lab_cos_seeds.txt`: 1 / 3 /
  10 seeds cosine 0.81777 / 0.82151 / 0.82277 vs dot 0.81278 / 0.81643 / 0.81763) and saturates by ~10 seeds.
  TRAINING with cosine instead is worse alone (26 `--tune cos:5 / cos:10 / icos:5`: 0.79380 / 0.79464 / 0.80387 vs
  0.81750, rank corr 0.88-0.93): the gain comes from normalising dot-trained vectors, not from a cosine simulator.
  View `34_cosine/cos_view.py` (10 vector seeds; columns cosine, item norm, user norm, item-normalised dot): day 6
  alone **0.82277** -- the strongest single model so far (MF30 0.81921) -- rank corr 0.946 with MF30. On the bundle
  set + sqrtview: half B 0.84157 -> **0.84201 (+0.00044)**, eval-weighted 0.82477 -> **0.82520 (+0.00043)**,
  1-20 known clicks 0.81609 -> 0.81651; cos_share / cos_logit rank 8th / 9th by gain. Pooling variants of the
  cosine (`lab/cos_variants.py`: sqrt-set, idf, unit-vector history) all within 0.001 alone, rank corr >= 0.97.
  Not kept: per-user taste sharpness (`lab/user_sharpness.py`: split-half reliability of the clicked item's
  within-event percentile 0.072 vs null 0.008).
- **Cosine everywhere the vectors are used: the audience views gain the most** (`26_audience/audience.py --cosine`:
  unit item vectors in, every score a cosine; 5 vector seeds as 26_ensemble_hops; caches `34a_<name>_c7.npz`).
  Day 6 alone, cosine vs the dot versions: aud 0.82356, audc 0.82458 (dot 0.81602), audc05 **0.82584** (dot 0.81764),
  wb 0.82325, hop2 0.82487 (dot 0.81923), hop2c 0.82467 (dot 0.81380, +0.011), hop3 0.82414 (dot 0.81485), hop4
  0.82355 (dot 0.81167). All eight on bundle + sqrtview + cos: half B 0.84201 -> **0.84278**, eval-weighted 0.82520 ->
  **0.82569**. Not kept: cosine views of the IDF and clicks-only MFs (`34_cosine/cos_view.py --vectors ...`; alone
  0.81898 / 0.80095): half B 0.84194 / 0.84198, eval-weighted 0.82500 / 0.82518 -- the core cosine already carries it.
  **Total vs the 26_ensemble_bundle LightGBM feature set** (sqrtview + cos + 8 cosine audience groups;
  `lab/paired_ci.py`): plain +0.00160 [+0.00134, +0.00188], **eval-weighted +0.00125 [+0.00085, +0.00173]**, P = 1.00.
  Built by `34_cosine/build.sh` (24_ensemble/ensemble.py, LightGBM + XGBoost + MLP): `34_cosine_s0.csv`, **public 0.82389**
  (+0.00110 over 0.82279, 0.9x the evaluation-weighted prediction).
- **The true taste space looks ~8-DIMENSIONAL** (`lab/geometry.py`, day 6, 10 vector seeds of the dim-32 MF of days
  0-5, impression-weighted covariance of the item vectors). Eigenvalue shares: eight nearly equal at 0.089-0.097,
  then 0.052, 0.034, 0.031, 0.017, 0.011, and a flat floor of ~0.005 for the other 19 -- the signature of an isotropic
  8-dim truth plus noise in the spare dimensions. No shared mean direction (|mean| / mean |v| 0.008). Cosine after
  projecting onto the top k components: k = 6 0.80723, **8 0.82381**, 10 0.82323, 12 0.82246, 16 0.82274, 24 0.82279
  vs 0.82277 unprojected (1-5 / 6-20 / 21+ known clicks at k = 8: 0.7510 / 0.8052 / 0.8338). Removing the top
  components instead (all-but-the-top) or whitening is much worse (top 1 removed 0.81548, whitened 0.78391). Earlier
  dimension sweeps stopped at 16 (0.8147 vs 0.8175) with l2 / steps tuned for 32; dim 8-12 are being tuned
  (`vectors_pool.py --dim d --tune ...`, `--dim` now forwarded to the tune stage).
  Low-dimensional MFs (`logs/2026-10-27_26_tune_lowdim.txt`, `..._dim8_long.txt`; 3 seeds, days 0-5 -> day 6; dim 32 base
  0.81750): dim 8 base 0.79487, l2 0.01 0.79813, 6000 steps 0.80658, 12000 0.81063, 20000 0.81079, 12000 + l2 0.01
  0.81153; dim 10 / 12 with 6000 steps 0.81477 / 0.81508. Small MFs are step-limited and plateau below dim 32: the
  32-dim MF fits better, and the medium directions past the top 8 (0.052 / 0.034 / 0.031) carry signal. The usable
  form is post hoc: project the dim-32 vectors onto their strong directions.
- **Projecting the vectors onto their 8 strong directions before the cosine: the #1 feature** (`34_cosine/cos_view.py
  --proj 8`, 10 vector seeds, principal directions per day from that day's vectors; cache `34_cosproj8_c7.npz`). Day 6
  alone 0.82383 (cosine 0.82277). On the experiment-34 set (public 0.82389): half B 0.84278 -> **0.84388**, eval-weighted
  0.82569 -> **0.82687**, 1-20 known clicks 0.81689 -> 0.81828; cosproj8_logit ranks **1st** by gain. Paired: plain
  +0.00110 [+0.00085, +0.00132], eval-weighted +0.00118 [+0.00081, +0.00152], P = 1.00.
  Not stacking alone (`lab/aud_combo.py`): projection HURTS the audience scores (k=8 A-0.5B 0.82482 vs 0.82600
  unprojected) -- the audience averages already denoise the weak directions. Unit clicker tastes in the audience
  (`lab/aud_weighting.py`): A-0.5B 0.82600 -> 0.82695 alone (click-weighted 0.82055, sqrt(n) 0.82592).
- **Unit clicker tastes in the cosine audience views: a further gain** (`26_audience/audience.py --cosine --unit-clickers`,
  5 seeds, caches `34b_<name>_c7.npz`; day 6 alone aud 0.82455, audc 0.82546, audc05 0.82679, wb 0.82324, hop2 0.82508,
  hop2c 0.82561, hop3 0.82538, hop4 0.82492 -- each +0.0001 to +0.0014 over 34a). Swapped for 34a on the 34 set +
  cosproj8: half B 0.84388 -> **0.84422**, eval-weighted 0.82687 -> **0.82718**; paired +0.00034 [+0.00016, +0.00051] /
  +0.00031 [+0.00002, +0.00061]. Set `SET35` in `lab/bundle_args.sh`.
- **Other projection cut-offs add little on top of k = 8** (SET35 base 0.84422 / 0.82718): + cosproj10 (alone 0.82337)
  0.84432 / 0.82746, best feature 56th by gain; + cosproj12 (alone 0.82249) 0.84424 / 0.82731. Not kept.
  Projection also rescues the clicks-only MF alone: projected cosine of its 5 vector seeds 0.81430 vs 0.80095 plain
  cosine (IDF MF: 0.81798 vs 0.81898).
- **Soft projection: stronger alone, redundant on top of the hard one** (`lab/soft_proj.py`; item vector = top-8 principal
  part + clicks / (clicks + K) x the rest; day 6, 10 seeds): plain cosine 0.82277, hard k=8 0.82383, soft K = 2 / 5 /
  10 / **20** / 50 / 200: 0.82385 / 0.82472 / 0.82535 / **0.82564** / 0.82526 / 0.82420 (light users and popular
  candidates both gain over the hard cut). View `34_cosine/cos_view.py --proj 8 --soft 20` on SET35: half B 0.84422 ->
  0.84422, eval-weighted 0.82718 -> 0.82737; paired +0.00000 [-0.00012, +0.00012] / +0.00018 [-0.00007, +0.00043]. It
  becomes the #1 feature but only replaces the hard projection. Not kept.
  Audience from soft-projected vectors (`audience.py --cosine --unit-clickers --soft-proj 20`, caches 34c_*), day 6
  alone vs 34b: aud 0.82726 (0.82455), audc 0.82546 (0.82546), audc05 0.82765 (0.82679), wb 0.82467 (0.82324), hop2
  0.82420 (0.82508), hop2c 0.82473 (0.82561), hop3 0.82470 (0.82538), hop4 0.82416 (0.82492).
  The soft-projected audience views on SET35 only replace existing features (each becomes #1-#3 by gain): + aud 0.84421 /
  0.82740, + audc05 0.84419 / 0.82725, + wb 0.84422 / 0.82721 (base 0.84422 / 0.82718). Post-hoc denoising is saturated.
- **A low-rank penalty during MF training hurts** (26 `vectors_pool.py --tune lowrank:lam`: lam x |v - P8 v|^2 on the batch
  item vectors, P8 = projector onto the top-8 principal directions, refreshed every 100 steps; 3 seeds, days 0-5 -> day 6):
  base 0.81750; lam 0.05 0.80786; 0.2 0.81079; 1 0.80846 (rank corr 0.958-0.964). Like the dim-8 MF: the 8-dim structure
  helps only as post-hoc cleaning of a dim-32 fit.
- **User-side shrinkage in cosine space: small alone, redundant by construction** (`lab/user_side.py`, day 6, 3 seeds; user =
  unit(hv) + g x unit(u2), u2 = mean unit-clicker audience C = A - 0.5B of the user's known clicks, g = c / (n + c)). With
  the item vector as candidate: 0.82221 (c 2) ... 0.82336 (equal mix); with the candidate's audience C: 0.82706 ... 0.82762
  (equal mix; 0.82695 without u2). No n-dependence helps. LightGBM already sees hv . C (audc) and u2 . C (hop2c); not built.
- **Experiment 35 public 0.82520** (+0.0013 over 34; eval-weighted half B predicted +0.0015).
- **More vector seeds would not help the projected cosine** (`logs/2026-10-27_lab_proj_seeds.txt`, day 6 alone): 1 / 2 / 3 / 5 /
  7 / 10 seeds 0.81950 / 0.82193 / 0.82274 / 0.82345 / 0.82372 / 0.82383.
- **An 8-dim per-item refit adds nothing** (`lab/refit8.py`: each item's 8 coordinates + bias re-fitted from its own impressions
  of days 0-5 on the users' projected unit tastes, prior centred on the MF's projected vector; day 6, 1 seed). Projected
  cosine 0.81950; refit lam 1 / 10 / 100: 0.81641 / 0.81950 / 0.81951 (with the refitted bias 0.814-0.817). The MF's
  projected vectors already hold what the impressions say; projection only removes the noise around them.
- **Softmax temperature of the projected cosine's event-relative features: little to gain.** Column 0 of every view is scaled
  to std 3 (for the projected cosine ~9 x cos, vs a fitted logistic scale of 5.7). Rescaled copies as extra groups on SET35:
  x0.6 half B 0.84431 / eval-weighted 0.82741 (its share becomes the #1 feature), x1.6 0.84416 / 0.82722 (base 0.84422 /
  0.82718). Within noise.
- **Pooling in cosine space: the cosine of the mean is best** (`lab/pooling_cos.py`, day 6, 5 seeds; 1-2 / 3-5 / 6-20 / 21+ known
  clicks): cos of mean 0.82274 (0.7149 / 0.7672 / 0.8044 / 0.8322); mean of per-click cosines 0.81873; max cosine 0.75485;
  soft-max tau 5 / 10 0.81084 / 0.79539. Projected: 0.82380 / 0.81984 / 0.75800 / 0.81195 / 0.79508. Taste is one direction.
  Projection hurts users with 1-2 clicks (0.7096 vs 0.7149) and helps everyone else -- LightGBM has both cosines.
- **Averaging aligned seed VECTORS is worse than averaging their scores** (`lab/align_seeds.py`: 10 seeds aligned by orthogonal
  Procrustes to their running mean; relative residual 0.36 -- seeds are different fits, not rotations). Projected cosine at
  1 / 3 / 5 / 10 seeds: score average 0.81950 / 0.82274 / 0.82345 / 0.82383, aligned vector average 0.81950 / 0.82252 /
  0.82318 / 0.82351.
- **The item side is at its information limit, also with an 8-dim truth** (`lab/replica8_noise.py`, log
  `2026-10-27_lab_replica8_noise.txt`). Replica with the truth projected on 8 directions, calibrated at scale 1.19 (MF 3 seeds
  0.81736 vs real 0.8175). Day 6, production cosine form, item vectors mapped into the truth space: true history + true
  candidates 0.85100; est history + true candidates 0.84296 (history side costs 0.008, mostly 1-5-click users 0.837 -> 0.806);
  true history + est candidates 0.82012 (candidate side costs 0.031); est both 0.81224 (1 seed), 0.81538 (3 seeds). Per-item
  cosine(est, true): 1-3 clicks 0.56, 4-10 0.77, 11-30 0.88, 31+ 0.95; 3 seeds add only ~0.02 -- data noise, not fit noise.
  Refitting every candidate's 8 coordinates from its own impressions with the TRUE user tastes and prior around the MF
  (lam 0.3 / 3 / 30): 0.80123 / 0.81798 / 0.82227 vs 0.82012 for the MF candidates -- nothing beyond the prior. Denoising the
  item vectors is closed; a gain has to come from structure the replica does not contain.
- **Real vs replica feature lifts** (`lab/replica_gap.py`, log `2026-10-28_lab_replica_gap.txt`). Same features from each
  world's own clicks and own MF vectors of days 0-5 (3 seeds), day 6, small LightGBM: base (cos8, popularity, CTR,
  impressions, history length, repeat) trained on half A, half B AUC, each group added alone. Real base 0.82345, replica
  0.81929. Lift real / replica: propensity +0.0032 / +0.0024, event context +0.0048 / +0.0049, max/top-3 similarity
  +0.0000 / -0.0000, mean similarity + coherence +0.0004 / +0.0002, recent-5 and oldest-5 tastes ~0 / ~0 (again no order),
  **residual dims (the 24 directions the 8-dim projection drops) +0.0022 / +0.0001**, all groups +0.0099 / +0.0074. The
  real data is not purely 8-dim: the dropped directions carry real signal (the replica truth is 8-dim by construction).
  Real users are a little more predictable in propensity (+0.0008 gap). Multi-interest structure (max similarity) is absent
  in both.
- **The residual-direction cosine adds little on SET35** (`34_cosine/cos_view.py --proj 8 --resid`, cache
  `36_cosresid8_c7.npz`, log `2026-10-28_lab_cosresid8.txt`): day 6 alone 0.577; half B 0.84422 -> 0.84436, paired
  +0.00015 [-0.00000, +0.00028], eval-weighted +0.00021 [-0.00008, +0.00053]. The full cosine already carries most of what
  the dropped directions add in `lab/replica_gap.py`.
- **Where SET35 loses AUC, again** (`lab/where_lost.py`, half B): event-level AUC 0.696, within-event 0.834; within part
  alone 0.838, model 0.844; a perfect event part would give 0.911. Same picture as `lab/decompose.py` (event level at its
  ceiling, see "Event-level ceiling"); the replica oracle's event level is 0.733 with the true propensity known.
- **No discrete categories behind the 8 directions** (`lab/simplex_test.py`, 7,398 items with >= 31 clicks, whitened
  8-dim coordinates vs a Gaussian of the same covariance): k-means(8) mean cosine to the nearest centre 0.564 vs 0.566;
  excess kurtosis 0.47 on every PCA axis vs 0.07 (a scale mixture: item norms vary), max over directions 0.62 vs 0.39.
  The taste space is a smooth isotropic cloud; no cluster or corner estimator to build.
- **The small views combined** (`lab/extra_member.py`, log `2026-10-28_lab_small_combo.txt`): SET35 + temp06 + cosresid8 +
  cossoft20 + soft_aud: half B +0.00007 [-0.00007, +0.00023], eval-weighted +0.00035 [+0.00005, +0.00069]; adding
  soft_audc05 as well: +0.00005 / +0.00021. A real but small eval-weighted gain (~+0.0003 public); kept for the next build
  only together with something larger.
- **Contrastive taste adds nothing** (`lab/contrast_taste.py`, day 6, 8-dim projected MF vectors of days 0-5, 3 seeds; taste =
  mean clicked - beta x mean of the 9 losers of each clicked event - gamma x mean shown in no-click events). beta/gamma 0:
  0.82260 (1-5 known clicks 0.7493); beta 0.2 0.82263 (0.7496); beta 0.5 0.82185; beta -0.2 0.82212; gamma +0.5 0.82147,
  -0.5 0.81902. The losers' information is already in the item vectors.
- **Taste strength grows with user activity on the real data, not on the replica** (`lab/replica_cells.py`, log
  `2026-10-28_lab_replica_cells_r8.txt`; each world's own MF vectors of days 0-5, 3 seeds, projected cosine on day 6, cells of
  user known clicks x candidate clicks). AUC real vs replica: 1-5 known clicks 0.705-0.759 vs 0.715-0.793 (-0.01 to -0.034),
  6-20 about equal, 21-50 +0.009 to +0.022, 51+ +0.000 to +0.043 (rare candidates +0.043); clicked-item coherence with the
  taste follows (1-5 -0.03, 51+ up to +0.05); top/bottom-quintile click lift ~2x larger for heavy users on the real data.
  The replica gives every user the same taste norm; real heavy users are more taste-driven and light users more random
  (plausibly: sharper tastes find more matches, click more, become heavy). Overall MF AUCs agree only on average. Repeat
  share of clicks real 0.12-0.13 vs replica 0.08-0.10 (no repeat bonus in this replica). Follow-up: MF impressions
  weighted toward heavy users (`--tune uw:-p`), the reverse of the earlier light-user weighting that hurt.
  Result (`logs/2026-10-28_26_tune_uw_heavy.txt`, 3 seeds, days 0-5 -> day 6): base 0.81750; uw:-0.3 0.81764; uw:-0.6
  0.81584; uw:-1.0 0.81058. Equal weighting is about optimal in both directions; the activity effect is LightGBM's to use.
- 12-dim-truth replica (`replica_sim.py --proj 12`): MF 0.80042 at scale 1.19, 0.82405 at 1.35 (real 0.8175 -> ~1.31).
- **SET35's score spread is right in every activity group** (log `2026-10-28_lab_slope_by_activity.txt`, half B, logistic
  recalibration of the model logit per known-clicks group): slope 0.85 / 0.91 for 0 / 1-2 clicks (0.2% / 1% of rows), 0.97-1.03
  for all others; per-group recalibration in-sample 0.84422 -> 0.84425. LightGBM already uses the activity-dependent taste
  strength through history length.
- Replica rounds with taste norm x (user clicks / median)^p, 12-dim truth, repeat bonus on (`lab/replica_round.sh`,
  `replica_sim.py --act-pow`): logs `2026-10-28_lab_replica_round_p12a{15,30}.txt`.
- **The mean of the clicked vectors is the best taste estimate we have, also against the truth** (`lab/replica_user_est.py`,
  8-dim replica scale 1.19, TRUE item vectors so only the user side is estimated; day 6, taste term only). Oracle theta
  0.86550 (1-5 known clicks 0.8827); unit mean of the clicked vectors 0.85602 (0.8429); contrast with the losers beta 0.3 /
  1.0 0.85537 / 0.85003; per-user MAP of the softmax-with-outside-option likelihood of their days 0-5 events (prior var 2,
  damped Newton; cos with the true taste falls from 0.82 at the mean start to 0.655) 0.80865 (0.6764). The user-side gap
  (0.010 overall, 0.040 for 1-5 clicks) is information, not pooling. Caveat: the replica's truth is itself built as a mean
  of each user's real clicks, which flatters the mean. Answers the "user side or item side" question: both are at their
  limits; item side costs 0.031, user side 0.008 (lab/replica8_noise.py).
- **Lambdarank over groups of many events beats binary on SET35 and SET36** (`lab/global_rank2.py`, logs
  `2026-10-28_lab_global_rank2_set3{5,6}.txt`; random groups of G events mixing days and users, early stopping on AUC).
  SET35: binary 0.84422 / 0.82718 (half B / eval-weighted); G500 0.84443 / 0.82752; G1000 0.84447 / 0.82761, paired +0.00025
  [+0.00012, +0.00038] / +0.00043 [+0.00018, +0.00070]; rank average with binary 0.84445 / 0.82753. SET36 (SET35 + the four
  small views): binary 0.84429 / 0.82754; G1000 0.84458 / 0.82777; rank average with binary 0.84456 / 0.82781. SET36 G1000
  vs SET35 binary: **+0.00036 [+0.00020, +0.00056] / +0.00059 [+0.00031, +0.00090]**. Experiment 36 builds it
  (`24_ensemble/ensemble.py` member `lgbrank_full`).
  **Follow-up (review challenge: was the MAP fit buggy?).** No: the likelihood reproduces the replica's generating
  probabilities (max error 1e-7), and for 98% of users with >= 20 events the MAP taste has a HIGHER log-likelihood than the
  true taste (-48.4 vs -51.9 per user): genuine overfitting at prior var 2. Matched priors (truth per-direction var ~0.37):
  centred at 0, var 0.37 / 0.1: 0.81329 / 0.81573 (unit); centred at c x the click mean, var 0.37 / 0.1 / 0.03: 0.83956 /
  0.84956 / 0.85393 -- monotone toward the mean (0.85602), never above. A LEAK flattered the mean in that comparison: the
  replica keeps the real warm-up clicks and builds the true taste from all real clicks, so a mean including them averages
  the items that define the truth. Fair version (`NO_WARM=1`, simulated clicks of days 0-5 only, log
  `2026-10-28_lab_replica_user_est_nowarm.txt`): oracle 0.86544; mean 0.82361 (1-5 clicks 0.7781); contrast beta 0.3 0.82367;
  best MAP (var 0.03 around c x mean) 0.82349; MAP centred at 0, var 0.1, unit 0.82127. The mean ties the best likelihood
  estimators on fair terms: resolved, no user-side estimator gain.
- Replica rounds with the default repeat bonus (log 4.2) are confounded (logs `..._replica_round_p12a15s111.txt`,
  `..._p12a30s102.txt`; MF 0.81890 / 0.82072): repeat-candidate click rate 0.50-0.68 vs real 0.29-0.31, so calibration
  lowers the taste signal and the real data beats the replica in almost every cell. The repeat-0 replica already matches the
  real repeat click rate (0.30-0.35): no repeat bonus beyond taste is needed. Rounds rerun with --repeat 0.
- Repeat-free replica rounds with activity-dependent taste (logs `..._replica_round_p12a15r0.txt`, `..._p12a30r0.txt`): MF
  0.83554 (scale 1.28, p 0.15) / 0.84526 (1.25, p 0.30), both above the real 0.8175 (recalibration needed, ~1.16 for p 0.15).
  p 0.30 overshoots (light users -0.04 to -0.05, heavy +0.03 to +0.05 vs real); p 0.15 is close by activity. What remains with
  p 0.15: real beats the replica for rare (0-10 clicks) and very popular (101+) candidates (+0.010 to +0.023, coherence +0.02 to
  +0.03) and trails for heavy users x mid-popularity (51+ x 11-30: -0.026). Repeat click rate: real flat at 0.29 across
  activity, replica rises 0.30 -> 0.45 (taste strength may not rise with activity for repeats); event click share rises
  more with activity in the real data (0.576 -> 0.672 vs 0.591 -> 0.641).
- **Systematic replica calibration** (`lab/replica_fast.sh` + `lab/replica_cells.py`, which now prints the overall projected-cosine
  AUC, real 0.82337, and a MISMATCH line: row-weighted rms of the cell AUC differences, rms of the repeat click rate and event
  click share differences by activity). 12-dim truth, act-pow 0.15, repeat 0, scale 1.16: item norms as estimated 0.80776,
  cell rms 0.0188 (mean +0.0164), repeat rms 0.0772; EQUAL item norms (`--item-norm 1`) 0.77606, cell rms 0.0495, repeat rms
  0.0339 -- the gap grows with item popularity (+0.05 to +0.085 for 31+ click candidates). The MF's longer vectors for popular
  items are real; equal-norm truth rejected. Next: amplified norms (`--item-norm -0.5`) and scale ~1.25.
- **Calibrated replica, best so far: `replica_fast.sh p12a15s125`** (12-dim truth, act-pow 0.15, repeat 0, scale 1.25,
  item norms as estimated): overall projected-cosine AUC 0.82230 vs real 0.82337; cell AUC rms 0.0095 (mean +0.0020).
  Amplified norms (`--item-norm -0.5`) worse (rms 0.0134, overall 0.83429). Left over: 101+ click candidates +0.022 to
  +0.026 real for users <= 50 clicks; heavy users x mid-popularity -0.019; repeat click rate (rms 0.094) and event click
  share by activity (real 0.576 -> 0.672, replica 0.591 -> 0.643).
- **Repeats are taste-driven on the real data, but their level does not rise with activity** (`lab/repeat_taste.py`, log
  `2026-10-28_lab_repeat_taste.txt`): real repeat click rate by cosine quintile 0.12 / 0.20 / 0.27 / 0.39 / 0.47 (within-repeat
  AUC 0.679; replica 0.660), but by activity real 0.31 / 0.29 / 0.29 / 0.29 vs replica 0.29 / 0.32 / 0.39 / 0.45. Heavy users'
  stronger taste lifts new items, not re-clicks: a satiation-like repeat penalty growing with activity. LightGBM has
  rep_any / rep_count / rep_recency x hist_len to learn it, and repeats are 2.6% of rows: not a lever.
- **User tastes are not drawn from archetypes** (`lab/user_clusters.py`, log `2026-10-28_lab_user_clusters.txt`; 1,483 users with
  >= 50 clicks before day 6, taste = mean of unit 8-dim projected MF vectors, 96% of its variance is signal). k-means mean cosine
  to the nearest centre, real vs a Gaussian with the same covariance: k 4 0.4618 / 0.4643, 8 0.5801 / 0.5847, 16 0.6753 / 0.6743,
  32 0.7490 / 0.7503. Heavier tails than Gaussian (whitened excess kurtosis 0.54 vs 0.10, like the items: taste strength varies),
  no clusters. No archetype prior for light users.
- **Popularity-weighted pooling of the user's clicks: equal weights are best** (`lab/pop_pool.py`, day 6, projected cosine, 3
  seeds, click weights (clicks + 1)^a): a -0.5 0.82194, -0.25 0.82300, **0 0.82337**, +0.25 0.82297, +0.5 0.82170, +1 0.81620;
  the same order in every user and candidate-popularity group (101+ candidates included). The replica's popular-item gap is not a
  pooling-weight issue.
- **Per-user past predictability (a choice temperature?)** (`lab/user_predictability.py`, log `..._user_predictability.txt`;
  margin of each clicked item over its event's other 9 under the leave-one-out taste, days 0-5, in-sample MF vectors). Split-half
  reliability is low (0.15 / 0.24 / 0.28 for users with 10-20 / 21-50 / 51+ scored events) and it correlates 0.68-0.88 with
  coherence. Day-6 projected-cosine AUC by its tercile within known-clicks bands: 6-20 0.7996 / 0.8052 / 0.8188 (coherence
  0.8035 / 0.8096 / 0.8107), 21-50 0.8250 / 0.8271 / 0.8300 (coherence 0.8254 / 0.8270 / 0.8301), 51-100 0.8349 / 0.8376 /
  0.8401 (coherence 0.8313 / 0.8408 / 0.8410). A slightly wider spread than coherence for light users only; candidate feature,
  not yet tested in LightGBM.
- **Return delay carries no click propensity** (review question; log `2026-10-28_lab_return_delay.txt`). Fit analogue of an
  evaluation event: each user's first event at/after the start of day d (d = 3-6, 177,185 events), delay = hours since the day
  start. Event click share by delay quintile (3.1 / 8.7 / 14.1 / 20.5 / 46.4 h): 0.625 / 0.633 / 0.635 / 0.629 / 0.634. Consistent
  with activity being independent of propensity: corr(log train events, click rate) -0.006 for users with >= 20 events (click
  rate by events 1-2 0.636 ... 51+ 0.625), while the click rate's own split-half reliability is 0.586.
- **First events do not explain 36's shortfall** (review hypothesis; `lab/paired_ci.py --first`, log
  `2026-10-28_lab_first_event_paired.txt`; only each user's first day-6 event, 17,597 of 48,976 half-B events, weights
  recomputed to evaluation.csv's mix). Eval-weighted B - A, all events vs first events: 36 change (SET35 binary -> SET36
  lambdarank G1000) +0.00059 vs **+0.00064** [+0.00019, +0.00107]; views only +0.00035 vs +0.00032; ranking only +0.00024 vs
  +0.00032; the views that transferred to public, 34's cos +0.00043 vs +0.00055 and 35's cosproj8 +0.00118 vs +0.00135. First
  events rank changes like all events. Likelier causes of 36's +0.00006 public over 35 s0: in the build the ranking objective
  replaces only one of three blend members (its gain is diluted), plus public noise between single builds (~0.0002).
  `lab/extra_member.py` and `lab/global_rank2.py` now print "first" on every run, and `TEST_DAY=5` moves the split to day 5.
- **Click propensity does not drift** (log `2026-10-28_lab_propensity_drift.txt`): click-rate reliability between interleaved
  halves of a user's pre-day-6 events vs time halves 0.581 / 0.581 (>= 20 events), 0.715 / 0.709 (>= 40). Day-6 event-level AUC
  of the user's past rate: plain 0.6333; recency-weighted with half-life 50 / 20 / 10 / 5 events 0.6331 / 0.6318 / 0.6283 /
  0.6199. Propensity is a fixed per-user constant.
- **Users do not differ in re-click habit** (`lab/user_repeat_rate.py`, log `2026-10-28_lab_user_repeat_rate.txt`). Of events
  before day 6 showing a repeat candidate, 32.6% had a repeat clicked; per-user split-half reliability of that rate 0.125 (2,229
  users with >= 10 such events). On half-B repeat rows (12,774, click rate 0.29, SET36 binary AUC 0.7191), logistic
  recalibration with the user's history repeat share / shrunk repeat-click rate / count: held-out all rows 0.84585 -> 0.84586 /
  0.84587 / 0.84585. Nothing per-user in re-clicking.
- **The SET36 model's residuals have no item, user or time structure on day 6** (log `2026-10-28_lab_residual_structure.txt`;
  half-B residual z-scores summed per key, correlation between two random halves of the events). Items (>= 5 / 10 rows per
  half, 3 splits): -0.005 to +0.011; users -0.014; hour of day -0.14 (24 keys); item residual vs the item's day-5-vs-earlier
  click-rate trend -0.08 (228 items). No item drift or appeal shift left to model. Residual TASTE directions per user (sum of
  residual x unit item vector, half 1 . half 2 vs user-permutation null): z +2.3 to +4.8 (8-dim), +3.0 to +5.2 (32-dim), +1.3
  to +2.5 (residual 24 dims) -- the expected signature of user-taste estimation error (consistent across a user's events), not
  usable without same-day labels; recency and contrast estimators already failed to reduce it.
- **Day-5 fresh holdout confirms the 36 changes** (review plan; `TEST_DAY=5`: LightGBM fit on days 2-4, day 5 halves; logs
  `2026-10-28_lab_d5_set35.txt`, `..._d5_global_rank_set36.txt`, `..._d5_paired.txt`). SET35 binary: half B 0.83919 /
  eval-weighted 0.82007 / first events 0.81884. SET36 binary 0.83932 / 0.82037 / 0.81912 (views: eval-weighted +0.00030
  [-0.00003, +0.00057], first +0.00027 [-0.00009, +0.00068]); SET36 lambdarank G500 0.83956 / 0.82051 / 0.81926; **binary +
  G500 rank average 0.83954 / 0.82057 / 0.81932: +0.00035 [+0.00016, +0.00051] / +0.00050 [+0.00021, +0.00074] / first +0.00047
  [+0.00013, +0.00087]**. Same direction as day 6, about 80% of the size. Built as 37 (`37_rank4/build.sh`: binary LightGBM kept,
  G500 ranking member added, fixed 2200 ranking trees via RANK_TREES to skip the early-stopping pass).
- **The per-user predictability view adds nothing** (`lab/predictability_view.py`, cache `37_predict_c7.npz`: trust-weighted
  projected cosine, shrunk margin, rank percentile, count, as of each day from W_d; log `2026-10-28_lab_predict_view.txt`). On
  SET36: half B 0.84429 -> 0.84433, eval-weighted 0.82754 -> 0.82744 (-0.00010 [-0.00035, +0.00011]), first events -0.00020
  [-0.00050, +0.00007]. Coherence and history length already carry it.
- **Leaderboard on 2026-10-29: 31st of 55 at 0.825353 (37c); 15th place 0.829516 (+0.0042), 1st 0.832966 (+0.0076).**
  Seed-2 build of 37 stopped (laptop slept on battery; worth ~+0.00001).
- **Bagged MF seeds are worse** (26 `vectors_pool.py --tune boot / boot:exp`, days 0-5 -> day 6, logs
  `2026-10-29_26_tune_boot.txt`, `..._bootexp.txt`). Idea: the replica says the item side loses to DATA noise, and our seeds
  differ only in init and batch order, so a seed average over bootstrap resamples would approximate the posterior-mean
  score. Poisson(1) event weights + resampled click-log examples, 6 seeds: 0.81530 vs 0.81847 (rank corr 0.980);
  Bayesian bootstrap (Exponential(1) weights, all events kept), 12 seeds: 0.81650 vs 0.81887 (0.988). Resampling noise
  is not averaged away; the MAP fit on all data is better.
- **REOPENED: the core MF's settings were tuned for the WRONG score** (2026-10-29; 26 `vectors_pool.py --tune` now also prints
  the day-6 AUC of the projected cosine, top-8 directions, unit vectors, averaged over seeds -- the form of SET35's #1 feature).
  Click-log weight 0.1 and 3000 steps were picked on 2026-10-20/22 by the MF's dot-product AUC; the projected cosine prefers
  more of both. Days 0-5 -> day 6, 3 seeds (`logs/2026-10-29_26_tune_proj8_sweep.txt`), dot / proj8: base 0.81750 / 0.82309;
  l2 0.03 / 0.08 / 0.12 / 0.2: 0.82320 / 0.82277 / 0.82215 / 0.81982; steps 2000 / 4500 / 6000: 0.82207 / 0.82341 / 0.82344;
  click-log weight 0.05 / 0.2 / 0.4: 0.82080 / **0.82382** / 0.82316; init std 0.05 0.82274; dim 64 0.82303, dim 16 0.82219.
  6 seeds (`..._proj8_aux.txt`, `..._proj8_aux2.txt`): base 0.81847 / 0.82400; aux 0.2 0.81709 / 0.82521; aux 0.3 0.81466 /
  0.82531; **aux 0.3 + 4500 steps 0.81376 / 0.82586 (+0.0019)**; aux 0.3 + 6000 0.82598; aux 0.5 + 4500 0.82579; aux 0.5 +
  6000 0.82601; aux 0.4 + 5000 0.82594; aux 0.3 + 4500 + l2 0.03 0.82587. A plateau from aux 0.3 / 4500 steps: the dot AUC
  falls while the direction quality rises (the click-log softmax trains directions, the impression BCE trains norms and
  biases). Experiment 38 builds the view from 10 vector seeds at aux 0.3 / 4500 steps (`38_aux_proj/vectors.sh`) and screens
  it on SET36 (`38_aux_proj/screen.sh`, day 6 and the fresh day-5 holdout, since day 6 also chose the setting).
  (`--loss choice` is not passed to the tune stage, so that row repeated base; untested.)
- **Experiment 38 screen: the better projected view keeps only ~1/8 of its gain after LightGBM** (`38_aux_proj/screen.sh`,
  logs `2026-10-29_38_screen_d6.txt` / `_d5.txt`; view `data/cache/38_cosproj8_c7.npz` from 10 vector seeds at aux 0.3 /
  4500 steps). Paired vs SET36, plain / eval-weighted: day 6 ADD +0.00023 [+0.00008, +0.00039] / +0.00016 [-0.00010, +0.00046];
  day 6 REPLACE cosproj8 +0.00027 [+0.00011, +0.00044] / +0.00026 [-0.00004, +0.00053]; day-5 fresh holdout ADD +0.00020
  [+0.00004, +0.00040] / +0.00014 [-0.00016, +0.00044]; REPLACE +0.00022 [+0.00006, +0.00040] / +0.00014 [-0.00012, +0.00042].
  Real on both days (plain intervals exclude 0) but ~+0.0002: the +0.0019 of the view alone is mostly already supplied by the
  other views (cosine, audience, sqrt, IDF) that LightGBM combines. Replacing beats adding. Below the bar alone; keep the
  replacement for the next build together with other changes.
- **The new vectors help only the projected view** (`logs/2026-10-29_38_cos_audience.txt`, `..._38_screen2_d6/d5.txt`). The
  unit-clicker cosine audience views rebuilt from the aux-0.3 / 4500-step vectors (5 seeds, `38b_*`), day 6 alone vs 34b:
  item side worse (aud 0.81786 vs 0.82455, audc 0.82435 vs 0.82546, audc05 0.82348 vs 0.82679, wb 0.82021 vs 0.82324),
  user side better (hop2 0.82561 vs 0.82508, hop2c 0.82645 vs 0.82561, hop3 0.82625 vs 0.82538, hop4 0.82590 vs 0.82492);
  plain cosine worse (0.81626 vs 0.82249). Swapping the four hops in as well as cosproj8: vs cosproj8 alone day 6 -0.00003 /
  -0.00003, day 5 +0.00002 / +0.00002 (plain / eval-weighted). The whole reopen is worth ~+0.0002: kept for a later build.
- **Choice-loss MF as a projected cosine: worse** (`logs/2026-10-29_26_tune_proj8_choice.txt`; `--loss` is now passed to the
  tune stage). 6 seeds, dot / proj8: choice base 0.81010 / 0.81513; aux 0.3 0.81588 / 0.81974; aux 0.3 + 4500 steps 0.81294 /
  0.81890 -- vs the BCE MF's 0.82400 / 0.82586. Closed.
- **Full-softmax click-log loss as a projected cosine: worse** (`logs/2026-10-29_26_tune_proj8_full.txt`, 6 seeds, dot / proj8):
  aux 0.3 + 4500 steps sampled 0.81376 / 0.82586; + full softmax 0.78561 / 0.82429; aux 0.1 full 0.79917 / 0.82347; aux 0.3
  full 0.78499 / 0.81929. Impression-weighted sampled negatives stay. The MF-settings reopen is closed at aux 0.3 / 4500 steps.
- **Reopened user-activity + scoring-day features add a little on SET36** (`lab/activity_view.py`, caches `38_act_{a,b,c}`;
  `38_aux_proj/screen3.sh`, logs `2026-10-29_38_screen3_d6/d5.txt`). On top of the aux-0.3 projected view, plain /
  eval-weighted: day 6 +0.00010 [-0.00002, +0.00022] / +0.00014 [-0.00009, +0.00036]; day-5 holdout +0.00023 [+0.00010,
  +0.00037] / +0.00037 [+0.00013, +0.00060]. **Both changes vs SET36: day 6 +0.00037 [+0.00020, +0.00053] / +0.00041
  [+0.00012, +0.00070]; day 5 +0.00045 [+0.00028, +0.00065] / +0.00050 [+0.00019, +0.00080].** Below +0.001 but consistent on
  both days, the size of the 36 change; built as experiment 38 (`38_aux_proj/build.sh`, set `SET38`) to average with 37c.
- **Posterior-style averaging of the MF: nothing** (2026-10-30; 26 `vectors_pool.py --tune snap:K/every, sgld:std`, days 0-5 ->
  day 6, 4 seeds, aux 0.3 / 4500 steps, log `2026-10-30_26_tune_snap_sgld.txt`). Projected cosine: final model only 0.82538;
  mean over 6 trajectory snapshots 250 steps apart 0.82537 (rank corr 1.0000: the trajectory has converged, snapshots are
  the same model); + Langevin noise on the item vectors in that window, std 0.002 / 0.006: 0.82533 / 0.82485. With bagging
  (above), the posterior-mean route to better rare-item vectors is closed.
- **Seed-1 build of 38 stopped** (2026-10-30): a second model seed has added +0.00002-0.00003 public (35, 37), not worth a
  4 h build. Build time of 38: MLP ~40 min, binary LightGBM 16 min, lambdarank 100 min, XGBoost 40 min.
- **MF-derived views drift with the MF's training length** (`lab/adversarial_views.py`, log
  `2026-10-30_lab_adversarial_views.txt`). Every SET38 view column, each user's first day-6 event (reweighted to the
  evaluation's known-clicks mix) vs the evaluation rows: most AUC 0.48-0.52. Separating: act_a span of history 0.670 and the
  recent-click counts (legitimate: longer gaps), cosresid8 col 1 0.397 / col 2 0.445, deep 0.553. Their first-event medians
  climb steadily with the fit day, i.e. with the days the MF was trained on (cosresid8 col 0 day 2..6: 0.31 / 0.42 / 0.53 /
  0.67 / 0.78, evaluation 0.84; col 1 2.17 -> 1.29, evaluation 1.18; cosproj8 cols 1-2 rise slowly too): the test rows sit
  beyond the range LightGBM was trained on. Screened as day-wise z-scored MF score columns (`lab/daynorm_screen.sh`).
- **Day-wise z-scored MF scores: nothing** (`lab/daynorm_screen.sh`, logs `2026-10-30_lab_daynorm_d6/d5.txt`; extra_member.py
  `DAYNORM=z`). Paired vs SET38, plain / eval-weighted: day 6 +0.00003 / -0.00001, day 5 +0.00005 / +0.00008, all intervals
  span zero. The lab split (fit days 2-5, test day 6) already contains the same drift as the build, and LightGBM copes.
- **A LEAN SET38: 15 of the 36 view groups carry all of it** (`lab/lean_screen.sh`, extra_member.py `LEAN=`, logs
  `2026-10-30_lab_lean_d6/d5.txt`). Groups ranked by split gain in the full model, refit on the top k. Gain share: soft_aud
  0.53, cosproj8 0.22, cos_hop4 0.04, everything else <= 0.025; core MF + popularity + sequence 0.04 together. Eval-weighted,
  day 6 / day 5 (full 0.82794 / 0.82087): top 0 0.82015 / 0.81155, 3 0.82626 / 0.81905, 6 0.82711 / 0.82008, 10 0.82761 /
  0.82057, **15 0.82779 / 0.82100**, 20 0.82798 / 0.82070. Top 15 (169 instead of 337 columns) matches the full set on
  both days; 14 of the 15 groups agree between days: soft_aud, cosproj8, cos_hop4, clicks, cos_audc05, uvec09,
  clicks_unif, act_b, act_c, cosresid8, choice, uoff10, deep, sqrtview (+ cos_hop2c on day 6 / clicks_idf on day 5).
- **Training on later fit days only is worse** (`lab/minday_screen.sh`, extra_member.py `MINDAY=`, log
  `2026-10-30_lab_minday.txt`). Paired vs SET38 on fit days 2+, eval-weighted: day 6, days 3-5 -0.00010 [-0.00036, +0.00015],
  days 4-5 -0.00053 [-0.00089, -0.00018]; day 5, days 3-4 -0.00056 [-0.00082, -0.00028]. The weak-MF early days still pay as
  training rows.
- **LightGBM capacity on the lean set: saturated** (`lab/lean_lgb_screen.sh`, extra_member.py `LGBP=`, log
  `2026-10-30_lab_lean_lgb_d6.txt`; paired vs the lean default 0.84447 / 0.82779, plain / eval-weighted): 127 leaves +0.00000 /
  -0.00013; 255 leaves (min leaf 100) -0.00013 / -0.00023; feature_fraction 0.5 +0.00006 / +0.00006; 127 leaves + l2 10 +
  min leaf 400 +0.00014 / +0.00013; lr 0.01 +0.00015 / +0.00015 (2216 trees). Bigger trees lose, slower learning gains a
  sliver. The model is information-limited, not capacity-limited.
- **A NEURAL CHOICE MODEL beats LightGBM on the same features** (2026-10-30; `lab/choice_model.py`, features exported by
  extra_member.py `EXPORT=` on SETLEAN; logs `2026-10-30_lab_choice_d6/d5.txt`). One softmax per event over 11 options:
  candidate scores from an MLP (256-128, dropout 0.1) on the Gaussian-rank features, a "no click" score from an MLP on the 57
  event-constant columns; cross-entropy on which option was taken; early stopping on half A (best at epoch 6-8, ~1 min per
  epoch). Half B plain / eval-weighted / first vs the lean LightGBM:
  day 6 0.84490 / 0.82809 / 0.82711 vs 0.84447 / 0.82779 / 0.82662, paired +0.00043 [+0.00019, +0.00066] / +0.00030
  [-0.00009, +0.00070]; day 5 0.84013 / 0.82139 / 0.82012 vs 0.83976 / 0.82100 / 0.81969. Rank corr with LightGBM 0.994,
  so it also blends: equal rank average day 6 0.84504 / 0.82837 (paired plain +0.00057 [+0.00045, +0.00069]), day 5
  0.84033 / 0.82164 (+0.00057 / +0.00064). The first model type to beat LightGBM on its own features here; the per-row MLP
  member of the builds scored below LightGBM.
  Variants (logs `2026-10-30_lab_choice_v2_d6.txt`, `..._choice_full_d6_run.txt`, day 6): 3 seeds averaged 0.84508 / 0.82833
  (seeds alone 0.84475-0.84490); slate attention (one self-attention layer over the 10 candidate embeddings, no-click score
  also sees their mean) 0.84478 / 0.82805 with 1 seed -- no gain; the full SET38 (337 columns) instead of the lean 169, 3
  seeds 0.84503 / 0.82837 -- same; equal rank blend with the full-set LightGBM 0.84518 / 0.82855 (+0.00052 / +0.00061 over
  it). Built as experiment 39 (`39_choice/build.sh`: member `choice_full` of 24_ensemble/ensemble.py on SETLEAN) and averaged
  with 38b.
- **Leaderboard 2026-10-30: 34th at 0.825545 (39b); 15th 0.830441 (+0.0049), 1st 0.833064; 18 teams at >= 0.8303.**
- **The model is calibrated by idle time; late evaluation events are not mis-levelled** (log
  `2026-10-30_lab_idle_calibration.txt`). Hypothesis: evaluation events come up to 7 days after the cutoff with no events in
  between, so gap features could read as low propensity. 39b's mean row rank does RISE with the evaluation day within
  known-clicks groups (6-20 clicks: 0.497 on day 7 -> 0.523 on day 12). On day-6 half B (lean LightGBM), sum of predicted p /
  actual click share by hours since the user's previous event: <1 h 1.011, 1-6 1.001, 6-24 0.973, 24-48 0.973, 48-96 0.981,
  >96 h 1.022 (555 events), no previous event 0.975. No idle-time bias to correct (evaluation shares of those groups 0.004 /
  0.059 / 0.337 / 0.224 / 0.197 / 0.151 / 0.027).
- **One combined item vector (own projected + audience): small, and saturates at once** (`lab/geometry2.py`, logs
  `2026-10-30_lab_geometry2_d6.txt`, `..._iter_d6.txt`; day 6, 38_vectors aux 0.3, single-score AUC plain / eval-weighted,
  5 seeds): own unit P8 vector 0.82563 / 0.80768, audience unit(A - 0.5B) 0.82616 / 0.80812, count-weighted mixes
  unit(Z c/(c+K) + g aud K/(c+K)) 0.82607-0.82671, plain sum unit(Z + aud) **0.82683 / 0.80882** (best; click-count
  weighting adds nothing). Iterating to a fixed point (audiences recomputed from the combined vectors, 3 seeds, base sum
  0.82646): round 1/2/3 0.82652 / 0.82652 / 0.82651 (beta 0.5, g 1); beta 1 or g 2 lower. LightGBM already holds both parts
  (cosproj8, soft_aud), so a vector-level combination is not a new representation. Post-hoc reworking of the existing
  vectors is exhausted at ~0.8265 alone.
- **Choice-model settings are flat** (`lab/choice_sweep.sh`, log `2026-10-30_lab_choice_sweep_d6.txt`, day 6, 2 seeds each,
  half B / eval-weighted): 256-128 0.84497 / 0.82825, 512-256 0.84499 / 0.82821, 512-256-128 0.84496 / 0.82824, lr 5e-4
  0.84491 / 0.82808, dropout 0.2 (512-256) 0.84494 / 0.82802. Keep 256-128.
- **The MF is UNDER-TRAINED for the projected geometry** (2026-10-30; 26 `vectors_pool.py --tune`, days 0-5 -> day 6, 3 seeds,
  log `2026-10-30_26_tune_lowdim_proj.txt`; raw MF AUC / proj8 cosine). Dim 32, aux 0.3: 4500 steps 0.81206 / 0.82473, 9000
  0.80137 / **0.82523**, 9000 + l2 0.02 0.80023 / 0.82501 -- the raw score falls (the 24 weak directions overfit) while the
  projected score keeps rising: every MF setting so far was tuned on the raw score. Small MFs at lr 3e-3 are far from
  converged: dim 12 4500 / 9000 steps 0.80585 / 0.81200 -> 0.81103 / 0.82354 (top 8 of 12); dim 8 0.78237 / 0.78829 ->
  0.79768 / 0.80190 (+0.014 from doubling the steps). Follow-ups: dim 32 at 15k / 25k steps (`..._tune_long_proj.txt`),
  dim 8 / 12 at lr 0.01-0.02 (`lab/lowdim_lr.sh`).
  Results: dim 32 aux 0.3 at 15k / 25k steps 0.82508 / 0.82488 (aux 0.5 15k 0.82500) -- the projected score PEAKS near 9000
  steps (+0.0005 over 4500), then falls. Small MFs with a faster optimiser converge but plateau below the projected dim 32:
  dim 12 lr 0.01 9000 / 20000 steps 0.82336 / 0.82349; dim 8 lr 0.01 0.81639 / 0.82283. The data is not purely 8-dim;
  fitting 32 and projecting is best. Under-training is worth +0.0005 at most (logs `2026-10-30_26_tune_long_proj.txt`,
  `..._lowdim_lr.txt`).
- **Propensity has NO headroom, even with future information** (`lab/propensity_headroom.py`, diagnostic only, log
  `2026-10-30_lab_propensity_headroom.txt`). Day-6 half B, lean LightGBM; events re-levelled by a logistic model fitted on half
  of half B (event logit + a user propensity estimate), AUC on the other half: as is 0.84521; with the past click share
  0.84515; with the user's click share over ALL other events of days 0-6 (past and future) 0.84515 -- every known-clicks group
  unchanged. The model's event level already holds all the per-user rate information; the gap to the leaders is on the item
  (within-slate) side.
- **No row-order artifact in evaluation.csv** (log `2026-10-30_lab_row_order_check.txt`, diagnostic, no labels): mean
  within-event rank of 39b's score by file position 5.48-5.52 (5.5 = none); every event has 10 contiguous rows.
- **Choice model trained toward the evaluation mix: worse** (`lab/choice_model.py --eval-weight w`, events weighted by
  (evaluation share / day share of their known-clicks group)^w, 2 seeds, day 6, log `2026-10-30_lab_choice_evalweight_d6.txt`):
  w 0.5 0.84488 / 0.82818, w 1.0 0.84461 / 0.82777 vs unweighted 0.84497 / 0.82825 (half B / eval-weighted).
- **Small MFs at a higher learning rate, last setting** (log `2026-10-30_26_tune_lowdim_lr.txt`): dim 12 lr 0.02 9000 / 20000
  steps 0.82020 / 0.82012 (lr 0.01: 0.82336 / 0.82349). Closed: no small MF reaches the projected dim 32.
- **Slates are not taste-targeted** (log `2026-10-30_lab_slate_targeting.txt`; day 6, projected unit vectors of the days 0-5
  MF). Cosine(user taste, candidate), known clicks 1-5 / 6-20 / 21-100: non-clicked non-repeat candidates -0.014 / -0.021 /
  -0.024, popularity-matched random items -0.001 / +0.000 / -0.001, clicked items +0.342 / +0.387 / +0.418. The generator
  draws by popularity, not by taste; the shown set carries no taste information (the slight negative is the selection of
  the clicked item out of the slate).
- **Co-click PPMI-SVD item vectors are far weaker, also as a projected cosine** (`lab/ppmi_cos.py`, log
  `2026-10-30_lab_ppmi_cos.txt`, day 6): MF projected cosine (1 seed) 0.82025; PPMI-SVD dim 32 top 8 / 12 / 32 0.64360 /
  0.70442 / 0.74220 (rank corr with the MF 0.32-0.63); averaged with the MF it only drags it down (best 0.80482). Clicks alone,
  without the impressions, do not identify the taste space.
- **Choice-loss MF, regularised: still below the per-candidate loss** (`--loss choice --tune`, 3 seeds, days 0-5 -> day 6, log
  `2026-10-30_26_tune_choice_reg.txt`; proj8 cosine): aux 0.3 + l2 0.1 0.81900, + l2 0.2 0.81887 (bce aux 0.3 / 4500 steps:
  0.82473). The choice likelihood overfits faster (yesterday: 4500 steps worse than 3000) and learns little from no-click
  events. Closed.
- **Public work on this challenge** (web search 2026-10-30): two public CTR repos report 0.810 / ~0.81 public (tree blends,
  choice models, ALS/SVD features, event calibration); no leader has published. Our own repo is public on GitHub.
- **Choice model with item embeddings learned end to end: worse** (`lab/choice_model.py --emb 16`, log
  `2026-10-30_lab_choice_emb_d6.txt`, 2 seeds: candidate score += tau x cos(mean embedding of the known clicks, candidate
  embedding)): 0.84396 / 0.82713 vs 0.84497 / 0.82825 without. Jointly learned ids overfit; the separate MF is the better
  item learner (as with lab/nn_ids.py).
- **Blends of more member types saturate** (day 6 half B, rank means, plain / eval-weighted): full-set LightGBM 0.84466 /
  0.82794; + choice (3 seeds) 0.84518 / 0.82851; + a second choice on the full set 0.84528 / 0.82863; + attention choice
  0.84522 / 0.82856; + embedding choice 0.84503 / 0.82836. A third member type adds <= +0.0001.
- **EXPERIMENT 40: the projected view from vectors trained TWICE as long -- +0.0006 after LightGBM on both days**
  (`40_long_mf/vectors.sh`: 10 seeds, aux 0.3, 9000 steps; view `40_cosproj8_c7.npz`, day 6 alone 0.82605 (10 seeds);
  `40_long_mf/screen.sh`, logs `2026-10-30_40_screen_d6/d5.txt`). Lean set with the view swapped, paired vs the lean set,
  plain / eval-weighted: day 6 0.84447 -> 0.84505, **+0.00057 [+0.00041, +0.00076] / +0.00060 [+0.00032, +0.00090]**; day-5
  fresh holdout 0.83967 -> 0.84020, **+0.00053 [+0.00037, +0.00070] / +0.00061 [+0.00033, +0.00088]**. Twice the 38 vector
  swap (+0.00027), from a representation trained for how it is used (the raw MF score falls with the longer training, the
  projected one rises). Set `SET40`; built as `40_long_mf/build.sh` (lgb + xgb + choice on SET40).
- **Audience views from the 9000-step vectors** (`26_audience/audience.py`, 5 seeds of 40_vectors, log
  `2026-10-30_40_audience.txt`). Day 6 alone, new vs old: soft-projected aud (`--soft-proj 20`, cache 40c_aud) **0.82855**
  vs 0.82726 (34c) -- the strongest single score so far; unit-clicker cosine variants (40b_*): hop4 0.82635 vs 0.82492,
  hop2c 0.82642 vs 0.82561, hop3 0.82618 vs 0.82538, hop2 0.82356 vs 0.82508, audc05 0.82016 vs 0.82679, aud 0.81296 vs
  0.82455 (the un-projected item-side audiences get worse with longer-trained vectors, as with 38's; the soft-projected one
  and the user-side hops get better). Screened on SET40 (`40_long_mf/screen2.sh`).
  Screened on SET40, day 6 (log `2026-10-30_40_screen2_d6.txt`), paired vs SET40, plain / eval-weighted: (a) soft_aud from
  the 40 vectors +0.00014 [-0.00001, +0.00032] / +0.00005 [-0.00026, +0.00036]; (b) + hop4 / hop2c +0.00012 / +0.00005.
  Noise after LightGBM despite the best alone score: the projected view already carries what the longer training adds. Not
  kept (day-5 run stopped).
- **OVERFITTING AUDIT on day 4, never a test day before** (`lab/audit_d4.sh`, log `2026-10-30_lab_audit_d4.txt`; fit days 2-3,
  day-4 halves; each set paired vs SET40, eval-weighted): SET35 -> SET40 **+0.00117 [+0.00065, +0.00159]** (plain +0.00104);
  SET38 (36 groups, 38 view) -> SET40 +0.00041 [+0.00006, +0.00074]; SETLEAN (38 view) -> SET40 +0.00044 [+0.00016,
  +0.00069] (the 40 view holds); without act_b / act_c +0.00002 [-0.00017, +0.00024] (nothing); without the 5 weakest groups
  (uoff10, deep, sqrtview, choice, cos_hop2c) -0.00008 (slightly better without). **The lab gains since 35 are real on an
  unselected day** (about as large as on days 5-6), so the poor public transfer since 35 (35 s0 0.82520 -> 38 s0 0.82525
  while day 4 says +0.0008 for SET35 -> SET38) is not selection on day 6: what helps on training-period events does not carry
  to the evaluation events. A distribution shift between the lab days and the evaluation period, not a split overfit.
  Simplification: SET40L = SET40 without those 7 groups (8 view groups), screened on days 6 / 5 / 4 (`lab/lean8_screen.sh`).
- **Day 4 predicts the leaderboard; days 5-6 were inflated** (2026-10-30): 40 alone public 0.82566 vs 38 alone 0.82525 =
  +0.00041, the exact day-4 paired gain of SET38 -> SET40 (+0.00041 eval-weighted) -- while days 6 / 5 said +0.0006 for the
  view alone. Day 4 is now the arbiter for build decisions (used only to confirm, never to screen).
- **SET40L (8 groups) loses** (`lab/lean8_screen.sh`, log `2026-10-30_lab_lean8.txt`), paired vs SET40 eval-weighted: day 6
  -0.00063 [-0.00091, -0.00033], day 5 -0.00055 [-0.00081, -0.00026], day 4 -0.00008 [-0.00037, +0.00027]. The 7 dropped
  groups are individually neutral but jointly worth a little; keep SET40 (15 groups).
- **cosresid8 from the 9000-step vectors: noise** (`40_long_mf/screen3.sh`, log `2026-10-30_40_screen3.txt`; day 6 paired vs
  SET40 +0.00007 / -0.00004). Not kept; day 4 not spent on it.
- **Learning-rate decay lets the MF train longer for the projected view** (26 `vectors_pool.py --tune ...+lrdecay[:cos]`, new
  option: linear or cosine decay to 0 over the steps; days 0-5 -> day 6, 3 seeds, log `2026-10-30_26_tune_lrdecay.txt`; raw
  MF AUC / proj8 cosine): constant lr 9000 steps 0.80137 / 0.82523; linear decay 9000 0.81308 / 0.82536; cosine decay 9000
  0.81290 / 0.82530; **cosine decay 13000 0.80846 / 0.82581** (+0.0006 over the 40 setting). The raw score also stays high:
  decay removes most of the late overfitting. Longer runs (`..._lrdecay2.txt`): cosine 18000 0.80236 / 0.82591, cosine 25000
  0.79538 / 0.82585, aux 0.5 cosine 18000 0.80024 / 0.82588 -- a plateau from 13000. Taken forward as **experiment 41**
  (`41_decay_mf/vectors.sh`: aux 0.3, 13000 steps, cosine decay, 10 seeds -> `41_cosproj8_c7.npz`; new `--lr-decay` flag),
  screened as a swap of SET40's projected view (`41_decay_mf/screen.sh`, days 6 / 5, then day 4 to confirm). **Result: no gain, closed.**
  The 10-seed view is only +0.0001-0.0003 alone per fit day (day 6 0.82628 vs 40's 0.82605; log `2026-10-30_41_vectors.txt`),
  and the swap LOSES after LightGBM, plain / eval-weighted: day 6 -0.00017 [-0.00030, -0.00004] / -0.00006; day 5 -0.00012 /
  -0.00010 (logs `2026-10-30_41_screen_d6/d5.txt`). Day 4 not spent. The 3-seed tune gain (+0.0006) was mostly a seed-count
  effect that 40's 10-seed average already had. A 20-seed view (40 + 41 vectors, `41_decay_mf/screen20.sh`, logs
  `2026-10-31_41_screen20_d6/d5.txt`) is flat too: day 6 -0.00004 / +0.00003, day 5 -0.00005 / -0.00001. **The projected-cosine
  channel is saturated**: more seeds, longer or decayed training no longer move LightGBM.
- **The lab test does not look like evaluation in TIME: evaluation users are rare visitors** (2026-10-31, `lab/eval_profile.py`,
  `lab/gap_profile.py`, logs `2026-10-31_lab_eval_profile.txt` / `_gap_profile.txt`; no evaluation labels exist or are used).
  Evaluation events run from day 7 to day 14 (median 7.70, 25% after 8.57, 10% after 9.93), lab test events span one day.
  Median hours since the user's last click: evaluation 42.6, day-6 first events 24.6; 20-click histories 39% vs 51%. No
  evaluation history click is at/after the cutoff, and every evaluation item appears in train. Item popularity goes stale
  only slowly and is weak anyway (`lab/staleness.py`: day-6 AUC of the cumulative item click rate frozen 1..5 days earlier
  0.5286 -> 0.5183). By hours since last click (eval share / day-6 share / event click rate / 40-lab AUC in bucket):
  0-6 h 4.7% / 42.3% / 0.65 / 0.856; 6-12 9.2% / 16.0% / 0.63 / 0.845; 12-24 18.3% / 17.7% / 0.63 / 0.842; 24-48 21.4% /
  12.6% / 0.61 / 0.839; 48-96 21.2% / 7.1% / 0.61 / 0.825; 96-192 16.0% / 3.1% / 0.58 / 0.801; 192 h+ or none 9.2% / 1.2% /
  0.54 / 0.756. **Half B reweighted to evaluation's gap mix gives 0.82626 for the 40 lab model -- public 0.82566 (40 s0)**,
  far closer than the history-weighted 0.82839. New "gap-weighted" metric in `lab/extra_member.py` and `lab/paired_ci.py`
  (SETLEAN -> SET40: plain +0.00057, eval-weighted +0.00060, gap-weighted +0.00082; public +0.00041).
- **Training weights matched to evaluation's gap mix HURT** (`lab/gapw_screen.sh`, `GAPW=p` in `lab/extra_member.py`; logs
  `2026-10-31_lab_gapw_d6/d5.txt`; paired vs SET40, plain / gap-weighted): p=1 day 6 -0.00086 / -0.00112, day 5 -0.00108 /
  -0.00118; p=0.5 day 6 -0.00046 / -0.00075, day 5 -0.00032 / -0.00039. Closed. **Why long gaps are hard: they are light
  users** (`lab/gap_profile.py` decomposition): mean known clicks fall from 47.7 (0-6 h) to 2.8 (192 h+), within-event AUC
  0.846 -> 0.733, event click rate 0.65 -> 0.54; but inside a known-clicks group the gap barely matters (21-50 clicks: 0-6 h
  0.856, 24-48 h 0.853, 48-96 h 0.851). The evaluation shift is the light-user shift we already weight for, measured more
  finely by the gap; no stale-taste effect to fix.
- **Light-user item click rates: no signal** (`lab/light_ctr.py`, log `2026-10-31_lab_light_ctr.txt`). Item click rate from
  fit days < 6 computed over all users vs over the test user's own known-clicks group, day-6 AUC by group: 1-5 known clicks
  all 0.5291 / group 0.5142 / <=5-only 0.5157; 6-20 0.5242 / 0.5194 / 0.5053; 21+ 0.5291 / 0.5274 / 0.5042. Light users do
  not prefer different items; the group rates are only noisier (corr of log rates 0.22). Closed.
- **Bigger MFs with cosine decay: same plateau** (log `2026-10-31_26_tune_dim_decay.txt`, aux 0.3, 13000 steps, cosine decay,
  3 seeds, proj8 cosine on day 6): dim 64 0.82559 (l2 0.08: 0.82584), dim 48 0.82574 (l2 0.08: 0.82586) vs dim 32 0.82581.
  Every MF setting tried converges to ~0.8258: the MF item vectors are at their ceiling.
- **Ensemble-diversity screens** (2026-10-31, `lab/diversity_screen.sh` + `lab/avg_preds.py`; logs `2026-10-31_lab_diversity_d6/d5.txt`).
  Each candidate trained on the lab split and rank-averaged 50/50 with the SET40 LightGBM, paired vs SET40 alone, eval- /
  gap-weighted. W1 extra-trees LightGBM (feature_fraction 0.5, 127 leaves): day 6 +0.00005 / +0.00004, day 5 +0.00012 /
  +0.00016. **W2 shallow regularised LightGBM (15 leaves, min_data_in_leaf 2000, feature_fraction 0.3): day 6 +0.00013 /
  +0.00010, day 5 +0.00019 / +0.00016** (alone it ties SET40: +0.00003 / -0.00005 and +0.00016 / +0.00009). D38 (full 36-group
  set with the 40 view): day 6 +0.00016 / +0.00019, day 5 +0.00002 / +0.00003. Small but consistent for W2 only; the real
  build already averages XGBoost and the choice model, so the increment there will be smaller.
- **CatBoost member: no** (`lab/diversity_cat.sh`, `MODEL=cat` in `lab/extra_member.py`, depth 8 Logloss; logs
  `2026-10-31_lab_diversity_cat_d6/d5.txt`). Alone vs SET40 eval- / gap-weighted: day 6 -0.00069 / -0.00077, day 5 -0.00096 /
  -0.00123; 50/50 average: day 6 -0.00002 / -0.00004, day 5 -0.00016 / -0.00028. Closed.
  **W2 on day 4 (unselected holdout; log `2026-10-31_lab_diversity_d4.txt`): alone vs SET40 plain / eval- / gap-weighted
  +0.00027 / +0.00030 / +0.00035; 50/50 average +0.00024 / +0.00027 / +0.00031 [+0.00015, +0.00049].** Holds on days 6, 5
  and 4 -> built as experiment 42 (`lgbshallow_full` member in `24_ensemble/ensemble.py`).
- **A third LightGBM-variant member adds nothing** (log `2026-10-31_lab_diversity3.txt`; equal thirds vs SET40 + W2, eval- /
  gap-weighted): + W1 day 6 -0.00000 / -0.00001, day 5 +0.00002 / +0.00006; + D38 day 6 +0.00008 / +0.00011, day 5 -0.00003 /
  -0.00001. Tree-variant diversity is used up after W2.
- **Stronger regularisation of the main LightGBM: within noise** (`lab/reg_sweep.sh`, logs `2026-10-31_lab_regsweep_d6/d5.txt`;
  single models on SET40 vs the 63-leaf SET40 LightGBM, eval- / gap-weighted, day 6 | day 5): R1 7 leaves +0.00005 / +0.00006 |
  +0.00024 / +0.00007; R2 15 leaves min_data 5000 +0.00006 / +0.00000 | +0.00016 / +0.00008; R3 31 leaves +-0.00005 / -0.00008 |
  +0.00019 / +0.00015; R4 15 leaves ff 0.15 +0.00011 / +0.00003 | +0.00008 / +0.00010. All intervals span 0; no setting beats
  W2 (42's member). Replacing the main member is not supported; day 4 not spent. Closed.
- **Time since the last EVENT / recent visits / no-click streak: noise** (`lab/last_event_view.py` -> `43_act_d_c7.npz`,
  `lab/act_d_screen.sh`, logs `2026-10-31_lab_last_event_view.txt`, `_act_d_d6/d5.txt`). Alone vs "the event gets a click" on
  day 6: log gap 0.503, events 24 h 0.497, 7 d 0.499, no-click streak 0.453 (inverted 0.547). SET40 + act_d vs SET40, eval- /
  gap-weighted: day 6 -0.00011 / -0.00024, day 5 +0.00015 / +0.00019 -- the days disagree. Closed; user propensity already
  covers it.
- **vast.ai box (2026-11-01; Threadripper 7995WX, 192 threads, RTX 3090).** One LightGBM job on all 184 threads is NOT
  faster than the M4 (thread overhead); capped at 24-32 threads (`NTHREADS`, new in `lab/extra_member.py` and
  `24_ensemble/ensemble.py`) a SET40 lab run takes 242 s vs 15-20 min on the Mac, ~7 jobs side by side. Box reproduces the
  Mac (`tools/box_verify.sh`): half B 0.84511 vs 0.84505, paired +0.00006 [-0.00003, +0.00015] (thread-count noise, 1278 vs
  1078 trees); all box comparisons below are box vs box (`lab_preds_boxbase_d*`).
- **Learning rate 0.01: no for the main LightGBM, W2 holds again** (`tools/box_batch1.sh`, `tools/box_batch2.sh`, logs
  `2026-11-01_box_batch1/`, `_box_batch2/`; vs box SET40, eval- / gap-weighted, day 6 | 5 | 4): SET40 lr 0.01 -0.00001 /
  +0.00006 | -0.00014 / -0.00012 | +0.00021 / +0.00015 (days disagree); W2 at lr 0.01 alone +0.00002 / +0.00011 | +0.00009 /
  +0.00013 | +0.00031 / +0.00019; their average +0.00008 / +0.00016 | +0.00006 / +0.00010 | +0.00035 / +0.00028 -- no better
  than 42's SET40 + W2 at lr 0.02. Tree settings are exhausted. Seeds 1-2 of the 42 ensemble built on the box for a
  seed-averaged final (`tools/box_build42seeds.sh`).
- **Choice member weight 0.40 instead of 0.25** (`lab/choice_weight.py`, log `2026-11-01_lab_choice_weight.txt`; box lab
  predictions of SET40 LightGBM, W2 and the SET40 choice model (default setting), LightGBM : W2 : choice = 2(1-w)/3 : (1-w)/3 :
  w, paired vs w = 0.25, eval- / gap-weighted, day 6 | 5 | 4): 0.35 +0.00007 / +0.00007 | +0.00004 / +0.00002 | +0.00008 /
  +0.00010; 0.45 +0.00010 / +0.00011 | +0.00005 / +0.00002 | +0.00012 / +0.00016; 0.60 +0.00011 / +0.00012 | +0.00002 /
  -0.00004 | +0.00013 / +0.00019. Fixed at 0.40 for the next build (+~0.0001). Alone the SET40 choice model beats the SET40
  LightGBM on days 6 and 4 (box batch 3).
- **Core MF with 60 / 90 seeds: no** (`tools/box_batch4.sh`, `_batch4b.sh`: 26 vectors_pool.py --mf-only, 13's settings, seeds
  30-89 in 12 parallel blocks; `tools/box_mfscreen.sh`, logs `2026-11-01_box_mf_mf60.txt` / `_mf90.txt`; SET40 with the core
  MF averaged over all seeds, paired vs box SET40 = MF30). Day 6 eval- / gap-weighted: MF60 -0.00016 / -0.00016, MF90 -0.00014 /
  -0.00022; day 5 about equal (0.82154 / 0.82157 vs 0.82163). The core MF is at its seed plateau at 30. Closed.
- **42 seed-averaged** (2026-11-01): seeds 1 and 2 of the 4-member 42 ensemble built on the box (`tools/box_build42seeds.sh`,
  `submissions/42_shallow4_s{1,2}.csv`; Spearman between seeds 0.9995); `42s3_avg3seeds.csv` = mean of seeds 0-2,
  `42s3b_avg_42s3_39b.csv` = its equal mean with 39b (the 42b recipe). `15_more_seeds/average.py` now writes 10 significant
  digits (6 left 413k distinct values of 529k: ties).
- **Day 3, the never-used split, for the frozen shortlist** (2026-11-01, `tools/box_day3.sh`, `lab/ens_compose.py`, log
  `2026-11-01_box_day3/`; members on SET40 fit on day 2 only, whole ensembles composed with their build weights; plain / eval- /
  gap-weighted): A (40: lgb, xgb, choice) -> B (42: + shallow) -0.00003 / +0.00001 / -0.00003 (nothing, as on public); B -> C
  (choice 0.40) +0.00014 / +0.00013 / +0.00018, all intervals above 0; A -> C +0.00011 / +0.00013 / +0.00015. Members alone,
  eval-weighted: lgb 0.81294, W2 0.81345, xgb 0.81306, choice 0.81356 (the choice model is again the best single member).
  Built as 43 (table).
- **Choice member C3 x10 seeds on the GPU: flat in the ensemble** (`tools/box_choice10.sh`, `lab/choice_model.py --device cuda`
  new: 67 s per seed vs ~20 min on the CPU; log `2026-11-01_box_choicegpu/`). Alone eval-weighted day 6 / 5 / 4: 0.82872 /
  0.82176 / 0.81773 (C0 x2: 0.82863 / 0.82160 / 0.81732). In LightGBM 0.4 : W2 0.2 : choice 0.4, vs C0 x2, eval- /
  gap-weighted: day 6 -0.00003 / -0.00001, day 5 +0.00003 / +0.00006, day 4 +0.00011 / +0.00003. Not adopted. Choice settings
  C1-C3 (batch 3) also flat. Ensemble-level changes are exhausted (lab and public: 42b +0.00001, 43s3b +0.00001).
- **Choice member on the full 36-group set (D38, 337 features): small consistent edge** (`tools/box_choiced38.sh`, log
  `2026-11-01_box_choiced38/`; C3 x10 on the GPU; ensemble LightGBM 0.4 : W2 0.2 : choice 0.4, D38 vs SET40 choice, eval- /
  gap-weighted): day 6 +0.00007 / +0.00006, day 5 +0.00005 / +0.00005, day 4 +0.00002 / +0.00007. Ensemble-level (~+0.00005);
  folded into the next build only alongside a representation change.
- **Representation screens on the box, 21 MF variants: none clears +0.0005** (`tools/box_reptune*.sh`, log
  `2026-11-01_box_reptune/`; aux 0.3 / 13000 steps / cosine lr decay base, projected cosine on day 6, 3 seeds; base 0.82581):
  18000 steps + l2 0.08 0.82600, user offset 0.82597, linear decay 0.82587, dim 48 + l2 0.08 0.82586, l2 0.08 0.82584, aux 0.5
  0.82570, hdrop 0.2 0.82566, impr 0.5 0.82558, aux 0.2 0.82555, std 0.05 0.82548, logq 0.82541, impr 2 0.82512, uw 0.5
  0.82473, boot 0.82191, choice loss 0.81801, lowrank 0.05 0.80694; --neg uniform / clicks and --pool idf / sqrtset gave exactly
  the base (the CLI options are not passed to the tune stage, so untested). The MF family is at its ceiling.
- **The audience view from the 40 vectors passes days 6, 5 and 4** (`tools/box_aud40.sh`, log `2026-11-01_box_aud40/`; the lead
  left unfinished on 2026-10-30). (a) soft_aud <- 40c_aud, paired vs box SET40, plain / eval- / gap-weighted: day 6 +0.00005 /
  +0.00004 / +0.00021, day 5 +0.00022 / +0.00023 / +0.00028, day 4 +0.00029 / +0.00033 / +0.00023. (b) + hop4 / hop2c from the
  40 vectors: day 6 -0.00004 / -0.00009 / +0.00011, day 5 +0.00022 / +0.00036 / +0.00031, day 4 +0.00017 / +0.00010 / +0.00014
  (weaker, not kept). A representation change: built as experiment 45 (`45_aud40/build.sh`).
- **40d (audience from all 10 40-vector seeds) vs 40c (5 seeds): noise** (`tools/box_aud40d.sh`, log `2026-11-01_box_aud40d/`;
  alone day 6 0.82873 vs 0.82855; in SET40 eval- / gap-weighted day 6 -0.00001 / -0.00003, day 5 +0.00009 / +0.00009, day 4
  -0.00002 / +0.00013). 41c (decayed vectors) 0.82865 and 4041c (20 seeds) 0.82872 alone: no better. 45 keeps 40c.
- **More audience views on top of 45: both worse** (paired vs the 45 set, plain / eval- / gap-weighted). (c) + cos_audc05 <-
  40c_audc (`tools/box_aud40c2.sh`, log `2026-11-01_box_aud40c2/`): day 6 -0.00011 / -0.00025 / -0.00023, day 5 -0.00005 /
  -0.00001 / -0.00004, day 4 -0.00007 / -0.00005 / -0.00004. (d) + a second audience group from the 41c vectors
  (`tools/box_aud41x.sh`, log `2026-11-01_box_aud41x/`): day 6 -0.00005 / -0.00012 / -0.00014, day 5 -0.00013 / -0.00013 /
  -0.00017, day 4 -0.00008 / -0.00008 / +0.00003. One audience view is enough; 45 stays the candidate.
- **cossoft20 rebuilt from the 40 vectors, added to 45: noise** (`tools/box_soft40.sh`, log `2026-11-02_box_soft40/`; view
  `34_cosine/cos_view.py --proj 8 --soft 20` on the 10 40 vectors, day 6 alone 0.82728). Paired vs the 45 set, plain / eval- /
  gap-weighted: day 6 -0.00004 / -0.00004 / +0.00004, day 5 +0.00001 / +0.00003 / +0.00004, day 4 +0.00001 / +0.00001 /
  -0.00005. Not kept.
- **cos_hop4 alone rebuilt from the 40 vectors, in 45: worse** (`tools/box_hop40.sh`, log `2026-11-02_box_hop40/`). Paired vs
  the 45 set, day 6 plain / eval- / gap-weighted -0.00004 / -0.00013 / -0.00015; eval-weighted day 5 0.82191 vs 0.82186, day 4
  0.81722 vs 0.81733. Every view rebuilt from the 40 vectors except soft_aud is now closed (cosresid8, cossoft20, cos_audc05,
  hop4, hop2c, a second audience view).
- **Out-of-fold MF by USER: worse than by event** (`28_oof_mf/oof_mf.py --by-user`, `NOBU=1 lab/oof_check.py`; logs
  `2026-11-02_28_oofu_validation_mf.txt`, `2026-11-02_lab_oofu_check.txt`). A fold's users are dropped from its MF entirely, so
  no MF saw the scored user's later behaviour (the cause found for the by-event loss). MF alone only 0.785-0.797 per day
  (the user's own clicks no longer shape the item vectors, and the user bias is untrained). LightGBM, half B / eval-weighted,
  with the MF user bias removed in every variant: time-forward 0.83398 / 0.81755 (with the bias 0.83441 / 0.81785);
  out-of-fold days 2-5 0.83303 / 0.81644; days 0-5 0.83304 / 0.81651. Closed: the time-forward rule stays.
- **45s6 (6 build seeds) public 0.826063 vs 45s3 0.826069**: same model; kept as the final (lower variance on private).
- **Gap map: where the remaining AUC is** (`lab/gap_map.py`, log `2026-11-02_lab_gap_map_d6.txt`; 45 set, day 6 half B,
  eval-weighted 0.82847). Oracle bounds: perfect within-event ranking +0.125, perfect "does the event get a click" +0.071.
  Now: within-event AUC 0.8351, event level (mean logit vs any click) 0.580. A uniform lift of the clicked candidate's logit
  by +0.03 (within-event 0.8351 -> 0.8386) is worth +0.0041 global, i.e. **0.83 on public needs ~+0.0035 within-event AUC**
  (about six experiment-40-sized representation wins). The leverage sits with 6-20 known clicks (eval share 45%, within
  0.818; +0.1 logit there = +0.0060) and 21-50 (34%, 0.842; +0.0042), then 1-5 (16%, 0.770; +0.0025). Empty-history events
  (1.3%) rank below random inside the event on days 6 / 5 / 4 (within 0.453 / 0.464 / 0.481), but there are only ~210-260
  such events per day (~100 clicked in half B), so this is within noise; item popularity alone gives 0.49-0.56 on them.
  Lifting them to 0.53 is worth ~+0.0001. Not a lead.
- **Pair-level error decomposition (external review's "experiment 46")** (`lab/pair_errors.py`, log
  `2026-11-02_lab_pair_errors_d6.txt`; 45 set, day 6 half B, eval-weighted 0.82847).
  - Global AUC is 99.997% CROSS-event pairs: within-event pairs are 0.0034% of all positive/negative pairs. The within-event
    oracle in `lab/gap_map.py` helps only because the clicked candidate then also outranks other events' rows. Positives vs
    negatives of clicked events AUC 0.8316 (58% of negatives), vs no-click events 0.8241 (42%); no-click events all at the
    bottom would give 0.902. Event level (sum of probabilities) 0.698, max 0.649, logsumexp 0.690, mean logit 0.580 -- vs
    0.668 with the TRUE propensity alone and 0.733 for the replica oracle (true propensity and taste): at its ceiling.
  - Within-event AUC by known clicks: 0 0.453, 1-5 0.770, 6-10 0.799, 11-20 0.826, 21-35 0.838, 36-50 0.848, 51-100 0.852
    (eval shares 1.3 / 16.4 / 19.4 / 25.6 / 24.3 / 9.4 / 3.6%): a smooth rise with information, no segment out of line.
  - Misranked (clicked, unclicked) pairs: 66% are pairs where the unclicked candidate's projected taste cosine is far above
    the clicked one's (> 1.0 on the std-3 scale; misrank rate 0.95), 11% where the clicked one has the higher cosine. 7.4%
    are an unclicked REPEAT outranking the click (rate 0.77; repeats are near-calibrated, see repeat-position notes).
    Popularity: misrank rate falls from 0.24 (clicked item 0-3 clicks) to 0.14 (31-100). Median misrank margin 0.90 logits,
    near misses (< 0.1) 6.7%: errors are confident, not borderline. The model follows the estimated taste; the misses are
    clicks that our taste estimate says are unlikely -- consistent with the item-vector information limit, no new category.
- **Empty-history fix: tiny, consistent sign** (`lab/empty_hist_fix.py`, log `2026-11-02_lab_empty_hist_fix.txt`; fixed rule:
  inside events with no known clicks the candidates are re-ordered by log item clicks, event mean logit kept). Eval-weighted,
  45 set: day 6 +0.00000 [-0.00008, +0.00007], day 5 +0.00004 [-0.00003, +0.00014], day 4 +0.00010 [+0.00000, +0.00022].
  Worth ~+0.00005 at most; a post-processing option for a final file, not a lead.
- **Third vast.ai box (2026-11-03; EPYC 7532 64 cores, RTX 4090, 503 GB, $0.57/h).** LightGBM needs `libgomp1` on the
  pytorch image. The 45 build's members re-run as lab members on days 6/5/4/3 (`tools/box_night1.sh`, logs
  `2026-11-03_box_night1/`): lgb / W2 / xgb / choice (SET45, 3 seeds) eval-weighted day 6 0.82847 / 0.82861 / 0.82841 /
  0.82882; the 45 composition (0.2 / 0.2 / 0.2 / 0.4, `lab/ens_mix.py`) day 6 / 5 / 4 / 3 0.82902 / 0.82227 / 0.81800 /
  0.81429. All comparisons below are paired vs that composition, eval- / gap-weighted.
- **SPECTRAL taste view: helps the choice member, not the trees** (`lab/spectral_view.py` -> `47_spec_c7.npz`: for the 16
  strongest principal directions e_k of 40's item vectors, f_k = (u.e_k)(v.e_k), u = unit mean taste of the known clicks, v =
  the candidate's unit vector, 10 seeds; sign-invariant per direction, so comparable across days and seeds). Directions 1-8
  alone AUC 0.67-0.75 on every day, 9-16 all 0.54: the taste space is 8-dimensional. The projected cosine fixes the metric
  (top 8, equal weights); here the network learns it (`lab/choice_model.py --add`). Choice member alone, SET45 vs + spectral:
  day 6 0.82882 -> 0.82912, day 5 0.82185 -> 0.82203, day 4 0.81781 -> 0.81770, day 3 0.81384 -> 0.81350 (the gain grows with
  the training days). In the ensemble: +0.00018 / +0.00019, +0.00010 / +0.00012, -0.00001 / -0.00002, -0.00003 / +0.00005.
  As 16 raw LightGBM columns (`tools/box_night3.sh`, `extra_member.py` now takes multi-column caches): day 6 +0.00035, day 5
  +0.00003, day 4 **-0.00037 [-0.00063, -0.00011]**, day 3 -0.00001: no. Raw (unnormalised) variant `--raw`: weaker (choice
  day 6 0.82897, day 5 0.82197).
- **Choice member on D45 (SET38 + 40's projected and audience views, 337 features)**: alone day 6 / 5 / 4 / 3 0.82899 /
  0.82195 / 0.81767 / 0.81383 (SET45 0.82882 / 0.82185 / 0.81781 / 0.81384); 1024-512 / drop 0.3 / 5 seeds no better.
- **Two choice members, SP + D (H3, 0.2 each): positive on all four days** (`tools/box_hedge.sh`): day 6 +0.00019 /
  +0.00017, day 5 +0.00013 / +0.00011, day 4 +0.00002 / +0.00003, day 3 (never used) +0.00005 / +0.00009. S + SP +0.00010 /
  +0.00007 / +0.00001 / +0.00001; S + D +0.00009 / +0.00007 / +0.00003 / +0.00005; S + SP + D +0.00014 / +0.00009 /
  +0.00003 / +0.00005. The two choice members differ most (Spearman of the full builds: SP vs 45 choice 0.9974, D vs 45
  0.9963, SP vs D 0.9950). Choice total weight for H3 (`tools/box_hedge2.sh`), vs 0.4, eval-weighted days 3 / 4 / 5 / 6:
  0.5 +4 / +5 / +2 / +7, 0.6 +4 / +7 / +2 / +12 (x1e-5). Built as experiment 47 with the even split 0.5.
- **Variants around 47's choice pair, all within noise** (paired vs H3 = choice SP + D, eval-weighted, days 3 / 4 / 5 / 6, x1e-5):
  SP -> SP + SPECTRAL AUDIENCE view (`lab/spectral_aud.py` -> `47_specaud`: per-direction products of the user's taste with
  unit(A - B) of 45's soft audience; `tools/box_night5.sh`) +1 / +6 / +5 / +1; D -> D + spectral (`tools/box_night6.sh`)
  -2 / +2 / +9 / +5; a third choice member SP with slate attention (`--attn`, choice 0.4/3 each; `tools/box_night7.sh`)
  -4 / +1 / +3 / +1. Not kept: two choice members carry the gain, the rest is lean-rule noise.
- **Per-direction taste SPREAD for the choice member: noise** (`spectral_view.py --var` -> `47_specvar`, sd of the user's
  unit click vectors along each direction; `tools/box_night8.sh`). SP -> SP + spread in H3, eval-weighted days 3 / 4 / 5 / 6:
  -2 / +2 / +2 / 0 (x1e-5).
- **LEARNED SPECTRAL SCORE: strong alone, harmful in LightGBM** (`lab/spectral_score.py` -> `47_specw`: logistic weights on
  the 16 spectral products and their within-event contrasts, fitted time-forward on days 2..d-1; day 2 = projected cosine).
  Learned weights are stable: the 8 taste directions ~2.35 each, directions 9-16 0.99 -> 0.25 (a learned soft projection).
  Alone: day 3 0.81611 (top 8 0.80771), day 4 0.82151, day 5 0.82864, day 6 0.83230 (top 8 0.82353; soft_aud 0.82855). As
  one LightGBM group (`tools/box_night9.sh`), paired vs SET45, eval-weighted: day 3 **-0.00111**, day 4 **-0.00038**, day 5
  -0.00002, day 6 -0.00001. The time-forward weights change the column's meaning from training day to training day (day 2
  fixed, day 3 fitted on one day, ...), and the trees already have the information. Closed.
- **A second spectral view from the 41 vectors (decayed-lr MF) in the spectral choice member: noise** (`spectral_view.py
  --family 41` -> `47_spec41`, `tools/box_night10.sh`). SP -> SP + spec41 in H3, eval-weighted days 3 / 4 / 5 / 6: -4 / +2 /
  0 / -2 (x1e-5). One spectral view is enough.
- **47 needs all three trees** (`tools/box_lean.sh`; choice pair 0.25 each, eval-weighted days 3 / 4 / 5 / 6, x1e-5): 47 vs
  the 45 composition +9 / +7 / +15 / +26; dropping xgb -4 / +1 / -2 / -2, dropping W2 -2 / -4 / -2 / -2, lgb alone at 0.5
  -13 / -4 / -6 / -7.
- **A cross-event pairwise loss for the choice members changes nothing** (`lab/choice_model.py --pair 1`, `tools/box_night11.sh`,
  logs `2026-11-03_box_night11/`). Global AUC is 99.997% cross-event pairs and the softmax is trained per event, so the
  loss adds w x softplus(log p_neg - log p_pos) over every clicked vs unclicked row of a 512-event batch. Day 6: SP / D alone
  eval-weighted 0.82912 / 0.82899, identical to the no-pair members (predictions differ: rank corr 0.999, max diff 0.1);
  47 with both replaced -0.00000 [-0.00007, +0.00005], gap-weighted +0.00000. The per-event softmax with its no-click
  level already ranks across events as well as the features allow. Stopped after day 6 (w 0.3 not run).
  At w = 10 (pair loss dominant; `tools/box_night12.sh`): members alone day 6 SP / D 0.82906 / 0.82897 (no pair 0.82912 /
  0.82899), day 5 0.82211 / 0.82210; 47 with both replaced, eval-weighted x1e-5: day 6 -4 [-12, +3], day 5 +5 [-2, +13];
  added as two more choice members (4 x 0.125): day 6 0 [-5, +3], day 5 +4 [0, +8]. Noise both ways. Closed.
- **XGBoost pairwise ranking member (GPU twin of lgbrank): noise** (`lab/extra_member.py MODEL=xgb XGB_RANK=1000`, random
  1000-event groups, fixed 700 rounds because XGBoost's GPU ranking AUC overflows int32 on our sizes; `tools/box_night13.sh`).
  Alone day 6 eval-weighted 0.82822, day 5 0.82126; 47 + it at 1/8 (trees 1/8 each): day 6 +0.00001 [-0.00004, +0.00005].
  Stopped. The LightGBM lambdarank member (lab/global_rank2.py, lr 0.02, 4 days in parallel at 16 threads) had not finished
  after 3.3 h on the EPYC 7532 box when credit forced the box down: no verdict.
- **Item-item CF with cosine / shrunk cosine / BM25 normalisation: weak alone, nothing in the choice member** (external review;
  `lab/item_cf.py`, caches `48cf_{cos,coss,bm25}_c7.npz`, logs `2026-11-03_box_cf/`). Alone, days 2-6: cos 0.594 / 0.601 /
  0.608 / 0.614 / 0.619, coss and bm25 within 0.002; rank corr with MF30 0.32. Added to the spectral choice member (SPCF) in
  47: day 6 -0.00002 [-0.00007, +0.00002]. The LightGBM test did not finish. Same verdict as 26's raw / lift / log-lift
  item-item views: co-click overlap carries nothing the MF space does not.
- **47 with 8 build seeds is the same file as 47s6** (fourth box, EPYC 9654 / RTX 6000 Ada, $0.70/h: 45 trees seeds 6-8 in
  ~30 min at 16 threads each, SP / D45 choice seeds 6-7; `experiments/47_spec_choice/build.sh "0 1 2 3 4 5 6 7"` ->
  `submissions/47s8_spec_choice.csv`): Spearman vs 47s6 0.99999. Not uploaded; more seeds are done.
- **Combination and cross-event calibration of the final score: no headroom; lambdarank is worth ~+0.00005 next to a choice
  member** (review; `lab/combine_check.py`, log `2026-11-03_lab_combine_check.txt`; day 6 half B, SET36 binary LightGBM,
  lambdarank G500 / G1000, choice model 3 seeds, all on the same rows; eval-weighted, paired event bootstrap vs the rank
  average of binary + choice = 0.82844). Alone: binary 0.82754, lambdarank G500 0.82776, choice 0.82837. Adding lambdarank:
  bin .375 / r500 .125 / ch .5 +0.00004 [+0.00001, +0.00006]; bin .25 / r500 .25 / ch .5 +0.00006 [0, +0.00011]; r500 + ch
  (no binary) +0.00006 [-0.00005, +0.00016]; equal thirds -0.00009. The +0.0004-0.0006 of 36/37 was against a trees-only
  ensemble; the choice member already holds the cross-event ranking gain. Calibration: logit average -0.00000, probability
  average +0.00000, geometric event x share +0.00000 [-0.00009, +0.00010]; choice event level x binary within-event share
  -0.00063, the reverse -0.00036. The rank average loses nothing to calibration; the event and within-event parts cannot be
  taken from different members.
- **Spectral LOADINGS in the choice member: worse** (`lab/spectral_view.py --sq --k 8`: (v.e_k)^2 of the candidate and
  (u.e_k)^2 of the user's unit taste for the 8 strong directions, sign-free, 10 vector seeds; cache `48_specsq_c7.npz`;
  `tools/box2_sq.sh`, logs `2026-11-03_box2/`). SP + loadings vs SP alone, 3 seeds, eval-weighted: day 6 -0.00019 [-0.00034,
  +0.00002], day 5 -0.00016 [-0.00035, +0.00001]. Direction-specific prominence and taste concentration only add noise.
- **48's composition holds on SET45, days 6 and 5** (`tools/box2_check48.sh`, members via the new fast path
  `lab/lgb_export.py` on the exports -- its binary member reproduces the extra_member.py one, day 6 eval-weighted 0.82847;
  lambdarank `--rank 500 --trees 2200`). Binary .5 + choice SP .5 vs binary .375 + lambdarank .125 + SP .5, eval- /
  gap-weighted: day 6 +0.00004 [+0.00001, +0.00007] / +0.00003, day 5 +0.00005 [+0.00001, +0.00009] / +0.00004. Lambdarank
  alone matches binary (day 6 0.82855 vs 0.82847, day 5 0.82181 vs 0.82186). Small, consistent, built as 48.
- **No per-segment calibration headroom** (`lab/segment_blend.py`, log `2026-11-03_lab_segment_blend.txt`). SET40 trees and
  the 10-seed choice model (box lab preds), rank average .5/.5 as base; per known-clicks group or per hours-since-last-click
  group, a choice weight or an additive score offset chosen on day 6 (coordinate ascent, gap-weighted), scored on days 5 and
  4 only, eval- / gap-weighted vs base: hist weights day 5 +0.00002 / -0.00000, day 4 +0.00006 / +0.00004; hist offsets
  (day 6 +0.00013 gap-weighted) day 5 -0.00011 [-0.00023, -0.00002] / -0.00012, day 4 -0.00011 [-0.00020, -0.00000] /
  -0.00012; gap weights day 5 -0.00003 / -0.00001, day 4 -0.00000 / -0.00003; gap offsets chosen all zero. The global
  rank average is already calibrated across segments; per-segment tuning only fits day 6. Closed. (Also checked: the
  evaluation-period exposure / staleness question is already closed above, 2026-10-31, and evaluation histories already
  feed the click log.)
- **BOOSTED CHOICE MODEL: no gain** (`lab/lgb_choice.py`, `lab/lgbc_compare.py`, logs `2026-11-03_lab_lgb_choice/`). LightGBM with
  a custom conditional-logit objective (softmax over the event's 10 candidates + a no-click option at score 0; gradient
  p - y, hessian p(1 - p); output p_i comparable across events), vs binary LightGBM on the same lean features (169 columns,
  fit days 2..T-1, matched settings). Eval-weighted alone: day 6 lr .05 +0.00037 [+0.00005, +0.00074] (0.82774 vs 0.82737);
  day 5 lr .05 -0.00024 [-0.00058, +0.00016], lr .02 -0.00025 (first events -0.00042). Next to the 10-seed neural choice
  member (bin .25 / gch .25 / nn .5 vs bin .5 / nn .5): day 6 +0.00016 [+0.00008, +0.00026], day 5 +0.00001 [-0.00007,
  +0.00011] (gap-weighted -0.00008). Pooled ~0: the neural choice member already supplies the choice structure, and the
  binary trees' per-row fit is as good as the listwise one. `ensemble.py` keeps an `lgbchoice_full` member (off unless named).
- **Forensic audit of mechanisms, event level and clustering: the model already fits them** (`lab/forensic_audit.py`, logs
  `2026-11-03_lab_forensic/audit_sum_d6.txt`, `_d5.txt`; matched binary LightGBM on the lean set, half B; an earlier run used
  1 - prod(1 - p) as the event probability, wrong for exclusive clicks -- ignore `audit_d*.txt`). (1) Event click share
  actual / predicted (sum of p) by hours since last click x user-rate tercile: 0.94-1.06 in every cell on both days (the
  within-user rise of click share with time since the last click in `docs/dataset/numbers.json` is reproduced by the
  model); by slate candidates already in the history 0.97-1.07, repeat rows CTR 0.305 vs 0.299 predicted. (2) A separate
  event-level LightGBM on event aggregates (mean / max / min of all 169 features + the model's event logit, 2-fold by user):
  event-level AUC day 6 0.68392 vs 0.68419 implied, day 5 0.68514 vs 0.68485; re-levelled global +0.00002 / -0.00008
  (as `lab/two_stage.py` on 2026-10-26). (3) Paired intervals resampling users instead of events are the same width
  (gch vs binary day 6 [-0.00001, +0.00071] vs [-0.00002, +0.00072]; ~1.95 day-T events per user). No hidden mechanism
  found; the evidence pack for the review is `experiments/FORENSIC_2026-11-03.md`.
- **No short-term "mood" in click outcomes; the same-day user residual was an estimation artifact** (`lab/mood_decay.py`, log
  `2026-11-03_lab_mood_decay.txt`; question: does the per-user residual that `lab/residuals.py` found within a day (+0.052,
  across days +0.015) reset at UTC midnight, or decay with time, so that an evaluation event a few hours after the user's last
  training event could use it?). Consecutive events of a user, residual = "event has a click" minus the user's click share on
  OTHER calendar days. Raw correlation rises with the gap (0.035 at < 0.5 h to 0.11 at 1-2 days) with no step at midnight
  (same day vs across midnight at equal gaps: 4-8 h 0.0615 / 0.0605, 8-16 h 0.083 / 0.077). Removing the part both residuals
  share through the error of the estimated level (variance r(1 - r) / n): < 0.5 h +0.004, 1-2 h +0.006, 2-4 h -0.001,
  4-8 h +0.008 (se ~0.004); 8 h-2 d +0.016 to +0.036, too small to matter. The previous event's outcome moves the next
  event's residual by < 0.005 at every gap. No mood to carry across the cutoff. Closed.
- **No `README.txt` came with the data** (user, 2026-11-03), although the task text lists one: there is no organiser material
  on the generator beyond `docs/phase_1_instructions.md`.
- **KERNEL AUDIENCE: noise in LightGBM** (`lab/kernel_aud.py`, `lab/kernel_blend.py`, `lab/kernel_view.py`, `lab/kern_compare.py`,
  `lab/lgb_choice.py --extra`; logs `2026-11-03_lab_kernel_aud/`). The local version of the audience view: a row's score is
  the item's click rate among the users it was shown to before day d, each weighted by exp(kappa (cos(t_u, t_v) - 1)) (8-dim
  projected unit tastes, the shown user's own clicks on the item removed, shrunk to the item's rate with weight 3; MF of days
  < d, 3 seeds of 38_vectors). Alone on day 6 (plain AUC): kappa 2 / 5 / 10 / 20 0.710 / 0.728 / 0.657 / 0.579 vs aud cos
  0.826, own cos 0.825, item rate 0.528 -- nonparametric in 8 dims with ~140 impressions per item is mostly noise. In a
  logistic blend with the two cosines and the item rate (half A -> half B): +0.0008 with the user's own past impressions,
  +0.0004 without them. As 6 LightGBM columns (log-odds and event-relative, kappa 2 / 5 / 10, own impressions excluded) on the
  lean set, binary lr 0.05, eval-weighted vs the matched baseline: day 6 +0.00017 [-0.00012, +0.00043] (next to the choice NN
  +0.00009 [-0.00006, +0.00022]); day 5 -0.00016 [-0.00043, +0.00012] (with the NN -0.00010, gap-weighted -0.00024
  [-0.00044, -0.00005]). With the own impressions too, day 6 +0.00011. The linear audience already has what the item's
  impressions say. Closed.
- **BAYESIAN CHOICE MODEL: the largest step since 35** (`49_bayes_choice/`, logs `experiments/logs/49/`). Multinomial choice with a per-user
  outside option, x_u . y_i + b_i + a_u[cat_i] + r [repeat], exposure counts informing a_u; exact user softmax and Gamma-augmented item
  updates, both Newton proposals with MH. What mattered: (1) the start. Click-only SVD start: 0.76 after 800 states; alternating
  Newton with FIXED priors reaches 0.8058 as a point estimate in about 20 sweeps. (2) Fixed hyperparameters: sampling them 0.7968 vs
  0.802 (and MAP with free ones collapses to x = y = 0). (3) More chains: F2 alone on Jan24 0.8087 (2 seeds), 0.8139 (4), 0.8164 (8).
  Stacker on Jan24, rows before Jan25: F1 + F2 +0.00258 (2 seeds), +0.00281 (4), +0.00301 (8); F3 (Gaussian prior), F4 (16 categories) and
  the exposure profile failed the +0.0001 / 4-of-5 rule. A rank score alone in the stacker loses 0.003: members need P and log(P / P0).
  Public 0.829038 for 4 seeds, in line with the lab gain.
- **Category tree member adds to the Bayes stack** (`49_bayes_choice/cat_member.py`, `catcheck.py`; lean LightGBM + k=9 and k=16 category columns,
  each built from the test day's own prefix). Jan24 alone 0.82056 vs old ensemble 0.81926; on top of the 8-seed candidate +0.00086, 5/5 folds.
- **Point estimates carry most of the Bayes family gain** (Jan24 stack with existing + category tree: no family 0.82107, two 1-seed point estimates
  0.82284, F1 + F2 8 seeds 0.82310, both 0.82336). The model (joint per-user fit, exact softmax likelihood, outside option) matters more than the
  averaging. Rank above 8 fails (K12 0.778, K16 0.756, K32 0.719 alone on Jan23). Seeds saturate: 16 vs 8 +0.00008.
- **The Bayes-style signal works far better as member features than as stacker inputs.** Jan24 alone: category tree 0.82056 -> with the view 0.82424, category choice 0.82167 -> 0.82430, vs 0.82393 for the whole 49d stack. A second point-estimate grid (24 variants) gave nothing through the linear stacker (best pick +0.00002 on Jan24).
