## 2. What we've learned

### ✅ Changes that improved our score

| Experiment | What we changed | Public AUC |
|---|---|---:|
| 03 → 04 | Removed user embeddings that caused overfitting | 0.728 → 0.767 |
| 04 → 05 | Added more historical clicks to training | 0.767 → 0.777 |
| 05 → 06 | Averaged multiple model runs | 0.777 → 0.796 |
| 07 → 09 | Used more of each user's click history | 0.798 → 0.805 |
| 09 → 10 | Used other known clicks to improve item representations | 0.805 → 0.813 |
| 10 → 11 | Shared item representations between user history and recommendations | 0.813 → 0.816 |
| 12 → 13 | Reduced MF training from 5,000 to 3,000 steps | 0.816 → 0.818 |

### ❌ Approaches that didn't help

We've tested several ideas that produced no meaningful improvement:

- **Click timing and order:** Recent clicks, time of day and click sequences added little or no value.
- **More complex recommendation models:** Attention and Transformer-based approaches did not outperform our existing MF model.
- **Alternative model settings:** Further tuning of MF and LightGBM produced negligible gains.
- **Additional features and model blending:** Combining small improvements produced a gain of only 0.0003 AUC, which isn't enough to justify a new submission.

**Key takeaway:** Our current model is becoming difficult to improve through small adjustments. We need to test ideas that provide genuinely new information.

## 3. What is our biggest weakness?

The model struggles when users or items have little interaction history.

- Users with more historical clicks generally receive better recommendations.
- Rare items are harder to predict because the model has fewer examples to learn from.
- Approximately 1% of events have no user click history, and these score poorly.

This suggests that **improving recommendations for users with limited history could be a promising direction.**

## 4. What should we try next?

The notes identify the following unexplored opportunity:

**A popularity-based fallback for users with no click history.**

Other experiments should focus on addressing weaknesses in the current model rather than repeatedly adjusting settings that have already been tested.

Before running an experiment:
1. Start with Experiment 13's pipeline.
2. Test changes using the approved training/validation split.
3. Record the results and compare them against the baseline.
4. Only consider a new submission if the improvement is meaningful.

## 5. When is an improvement worth submitting?

Our current validation baseline is **0.83571 AUC**.

- **Less than +0.0002:** Likely noise.
- **Around +0.0003:** Too small to justify a submission on its own.
- **At least +0.001:** Our target for a meaningful improvement.

We have a limit of 10 submissions per team per day, so we should prioritise experiments that have a reasonable chance of improving the score.

## Bottom line

We've built a strong baseline by improving how we represent user preferences and item relationships.

The biggest gains have come from using more click history, improving item representations and averaging multiple model runs.

Most straightforward improvements have now been exhausted. Our next experiments should focus on **new sources of useful information and the model's weakest cases**, rather than more minor parameter tuning.