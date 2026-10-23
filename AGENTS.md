# AGENTS.md

Guide for coding agents and teammates working on this repo: what the best model is, how to
reproduce it, and how to try to improve it without fooling ourselves.

## Task in one paragraph
TechArena Ireland 2026, Phase 1 (`docs/phase_1_instructions.md`). Each event shows a user 10
candidate items; at most one is clicked. Predict a click probability for every row of
`data/evaluation.csv`; the score is **global ROC AUC over all rows**, so predictions must rank
well both inside and across events. 10 uploads per day, deadline 2026-11-04. The public
leaderboard uses part of the evaluation rows; the final ranking uses the private part.

## Best model (public AUC 0.81807)
`submissions/15_avg_13_15.csv` = mean of two runs of the same pipeline with different MF seeds:

| Run | Script | Output |
|---|---|---|
| 13 (MF seeds 0-14) | `experiments/13_retune_7days/retune.py` | `submissions/13_retune_7days.csv` (public 0.81784) |
| 15 (MF seeds 15-29) | `experiments/15_more_seeds/more_seeds.py` | `submissions/15_seeds15_29.csv` |
| average | `experiments/15_more_seeds/average.py` | `submissions/15_avg_13_15.csv` |

Both scripts are standalone (no imports from other experiment folders). Pipeline per cutoff day C
(days counted from 2025-01-20 00:00 UTC; C = 7 for the submission):
1. **Base features** (streaming `train.csv`, then `evaluation.csv`, in time order): as-of
   features per candidate and per event (repeat clicks, co-click similarity, transitions, item
   and user click rates, time of day, ...), plus a log of every click known before the cutoff.
2. **MF** (own process: torch and LightGBM crash in one process on macOS). History-only
   matrix factorisation, ONE vector per item, user vector = mean of the vectors of the user's
   known clicks strictly before the event (up to 100). Loss = impressions (binary cross-entropy)
   + 0.1 x click-log softmax (each logged click vs 9 impression-weighted negatives, context = the
   user's other clicks). dim 32, l2 0.05, lr 3e-3, 3000 steps, init std 0.005, 15 seeds averaged.
   Fit day d is scored by an MF trained on days < d; evaluation by an MF trained on all fit days.
3. **LightGBM** (binary, 63 leaves, lr 0.05) on base + MF features + each candidate compared
   with the other 9 in its event. Trees chosen by early stopping on day C-1, then refit on days
   2..C-1.

## Reproduce
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# data/train.csv and data/evaluation.csv from the portal (git-ignored)
.venv/bin/python experiments/13_retune_7days/retune.py --cutoff 7 --out submissions/13_retune_7days.csv
.venv/bin/python experiments/15_more_seeds/more_seeds.py --cutoff 7 --out submissions/15_seeds15_29.csv
.venv/bin/python experiments/15_more_seeds/average.py submissions/13_retune_7days.csv \
    submissions/15_seeds15_29.csv --out submissions/15_avg_13_15.csv
.venv/bin/python src/check_submission.py submissions/15_avg_13_15.csv
```
About an hour per run from scratch on a laptop (keep it plugged in; `caffeinate -is` on macOS).
Base features and MF scores are cached in `data/cache/`, keyed by a hash of the code and
settings, so re-runs only redo what changed. Deleting the cache is always safe.

`--cutoff 5` runs the local validation instead (first event per user at/after day 5, nothing
written): 13 scores **0.8136** there.

## Rules that must never be broken
These follow "History and time split" in `docs/phase_1_instructions.md`. Breaking them gives
local scores that do not hold on the leaderboard (experiment 02 did this once).
- No labels from test events. Click-rate statistics are frozen at the cutoff (inside the fit
  period they update once per day). Click histories are inputs and may be used as of each event.
- MF scores for fit day d come only from an MF trained on days before d.
- Every setting is chosen inside the fit period. Never tune on `evaluation.csv`, on the
  validation test events, or on leaderboard scores.
- Uploads are for finished candidates, not for testing ideas.

## How to iterate (one batch)
1. **Read `experiments/RESULTS.md` first.** Its "What we learned" section lists every idea
   already tried and its result; do not retry them without a new reason.
2. **Write the change as a fit-period check**, official split (cutoff 7), with fixed baselines:
   - LightGBM-level ideas (features, model settings, blends): train LightGBM on fit days 2-5;
     split day 6 in halves by event (`np.random.default_rng(0)`); half A for early stopping and
     blend weights, report AUC on half B. **Baseline: 0.83571.** Template:
     `experiments/lab/xgb_blend.py` or `experiments/lab/lgb_settings.py` (they reuse 13's
     cached arrays and run in minutes).
   - MF-level ideas: copy `experiments/13_retune_7days/retune.py` into a new folder and use
     `--cutoff 7 --tune base,<variant> --tune-seeds 3` (MF trained on days 0-5, AUC on day 6).
     **Baseline: 0.8175 with 3 seeds.** Always use at least 3 seeds: single-seed gains have
     vanished three times. To compare a blend fairly, include a control that averages the MF
     with itself on other seeds (see `experiments/18_sequence/sequence.py`, `seedoff:`), which
     gives 0.8183.
3. **Save the output** to `experiments/logs/YYYY-MM-DD_<experiment>_<what>.txt` (`| tee`;
   `.log` files are git-ignored) and add the result to RESULTS.md, kept or not.
4. **Only build a submission for a clear win: about +0.001 or more on half B.** +0.0002 is
   noise. Then add the change to a standalone copy of 13 as a new experiment folder
   `experiments/NN_name/` (next free number), run `--cutoff 7`, check the file with
   `src/check_submission.py`, and hand it over for upload. Record the public score in the
   RESULTS.md table.

### Conventions
- New experiments are standalone copies of the current best script; the module docstring
  says what changed, why, and the fit-period result. With the new option off, the script
  must reproduce 13 exactly.
- Quick checks on cached arrays go in `experiments/lab/` and are run from the repo root.
- Kill background jobs by exact PID (`pkill -f` matches shell command lines too).
- Do not import torch and LightGBM in the same process.

## What is known (details in RESULTS.md)
- The simulator behaves like: user taste (the SET of clicked items, no order, no recency) x item
  vectors + item popularity + a per-user click propensity. The current model covers all four.
- Validation first events lose AUC mainly on light users: 1-5 known clicks 0.75, 6-20 0.81,
  21+ 0.84. Their histories simply carry little information.
- Ruled out (all within noise or worse): recency/time/position/session effects, per-user
  embeddings, candidate-set and shown-item profiles, shortened-history training, item-vector
  smoothing, co-click pre-training, EASE item-item scores, similarity features, popularity
  taste, row weighting, a transformer over the click sequence, XGBoost blends, LightGBM and MF
  re-tuning (dim 16/24/64, steps, l2, lr, click-log weight). The three smallest gains combined
  (popularity taste + LightGBM lr 0.02 + XGBoost blend) give only +0.0003 on half B.
- More MF seeds keep helping slightly (10 → 15 → 30), but the gain per hour is tiny.

The next real gain probably needs a new source of information or a different kind of model,
not more tuning of this one.
