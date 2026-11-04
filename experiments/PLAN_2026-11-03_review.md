# Plan for review: last day before the deadline (2026-11-03)

For an outside reviewer: please scrutinise the approach, not just the code. Deadline 2026-11-04. Repo rules are in
`AGENTS.md`; every idea tried so far is in `experiments/RESULTS.md` ("What we learned").

## Where we are
- Metric: global ROC AUC over all evaluation rows (52,939 events x 10 candidates, at most one click each).
- Best uploaded: **45s3, public 0.82607**. Leaderboard (2026-10-30): 1st ~0.833, 15th ~0.8304. Gap to 15th ~0.004.
- Best built, not uploaded: **`submissions/47s6_spec_choice.csv`**. It is a rank average of three tree members (LightGBM,
  XGBoost, shallow LightGBM; 1/6 each) and two neural choice members (0.25 each). The choice members are SP (feature set
  SET45 plus the spectral taste view) and D (feature set D45, 337 features). Gain over the 45 composition, eval-weighted,
  x1e-5: day 6 +26, day 5 +15, day 4 +7, day 3 +9. Expected public about +0.0001-0.0002.
- Compute: vast.ai box (EPYC 7532 with 64 cores / 128 threads, RTX 4090), about $4.7 of credit (~8 h).

## Lab protocol (how every change is judged)
- Members are trained on fit days 2..T-1. Day T is split into halves A and B by event (`rng(0)`). Half A is used for early
  stopping; AUC is reported on half B.
- Comparisons are paired bootstraps (`experiments/lab/paired_ci.py`) in two views: evaluation-weighted (weights match the
  evaluation's known-clicks mix) and gap-weighted.
- Test days are 6, 5, 4 and 3. Day 3 was never used for selection.
- A change is kept only if it is positive on all four days. A submission is built only for about +0.001, or for a
  consistent smaller gain with no new tuned settings.

## Diagnosis that drives the plan
- Global AUC is 99.997% cross-event pairs.
- On day 6 half B, event-level AUC (mean logit) is 0.58 and within-event AUC is 0.835.
- Propensity has no headroom: adding the user's click share from all other events, past and future, adds nothing. The
  event-level part is at its ceiling.
- The remaining loss is on the item side. In a replica simulator, ~0.04 of AUC is lost to item-vector estimation error, and
  most of that is not recoverable from ~140 impressions per item.
- The neural choice member (softmax over 10 candidates + "no click") is the strongest member. Its loss only ranks candidates
  within an event plus a no-click level; it is never trained on cross-event pairs.
- The trees gained +0.0004-0.0006 (days 6 and 5) from lambdarank over random 1000-event groups, a cross-event objective.
  That member was dropped at experiment 40 when the feature set was trimmed, and was never re-tested next to the choice
  members.

## Running now (`tools/box_night11.sh`, logs `experiments/logs/2026-11-03_box_night11/`)
1. **Cross-event pairwise loss for both choice members** (`experiments/lab/choice_model.py --pair w`, w = 1 and 0.3).
   - Loss = softmax cross-entropy + w x mean softplus(-(log p_pos - log p_neg)), over every clicked row vs every unclicked
     row in a 512-event minibatch (any event).
   - 3 seeds per member; the 47 mix is rebuilt with both members replaced.
2. **Lambdarank LightGBM member** (`experiments/lab/global_rank2.py`, groups of 1000 random events, SET45, lr 0.02) added
   to 47 at weight 1/8. The trees drop to 1/8 each; the choice weights are unchanged.

Both are judged against 47 on days 6/5/4/3, paired, eval- and gap-weighted.

## Decision rules
- Positive on all four days and about +0.0005 or more eval-weighted: build it as experiment 48 (6 seeds; swap members into
  the 45 seed files where possible) and hand it over for upload.
- Smaller but consistent: build it only if it adds no new tuned setting. That holds for w = 1, and for adding the rank member
  at a fixed equal-ish weight.
- If w = 0.3 wins and w = 1 does not, treat it as tuning on lab days and require day 3 to agree.
- Otherwise: close it, record it in RESULTS.md, and recommend 47s6 as the final upload.
- Seeds 6-11 of 47 (expected ~+0.0001) are the fallback use of the box only if nothing better is queued.

## Questions for the reviewer
1. Is the pairwise term well posed? A 512-event batch gives ~320 positives x ~4800 negatives. Should negatives include
   rows of the same event (they do now), and is log p (rather than the logit s_k) the right score to rank?
2. Is the lab protocol leaking selection? Days 6/5/4 were used for earlier choices; day 3 is the only clean day. Is a
   +0.0001-sized day-3 result meaningful at ~49k half-B events?
3. Weights: 47's choice total of 0.5 was picked on lab days. Is adding a fifth member at 1/8 a new tuned setting?
4. Anything structural we have missed that could explain the leaders' +0.007? `RESULTS.md` rules out about 200 ideas,
   including:
   - sequence and recency effects
   - per-user embeddings
   - candidate-set information
   - ID structure (the IDs are not hashes of simple counters)
   - a propensity ceiling
   - stacking
   - CatBoost
   - more MF seeds
   - VAE / SASRec / EASE / PureSVD
5. The final pick under overfitting risk: is 47s6 (fixed equal weights, positive on the untouched day 3) the right file, or
   is 45s3 safer?
