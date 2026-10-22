# techarena2026-atu
Project for TechArena Ireland Hackathon 2026

# Links
Submissions for the .csv are made below. The Phase 1 instructions and train/val dataset can also be found in the link.
https://portal-techarena-ireland.hackathon.com/submit.html

# Submissions
Submissions can be made 10 times per day from October 21st to November 4th.

Current best: **`submissions/11_tied_items.csv`, public AUC 0.81584**. All scores and what changed
between experiments: [experiments/RESULTS.md](experiments/RESULTS.md).

# Team Members
- Mohamed Elmofti, L00170666@atu.ie
- Pawan Venugopal, L00213636@atu.ie
- Laurin Burge, L00213713@atu.ie
- Lam Xiang Le, L00187730@atu.ie
- Jana Stachova, L00194712@atu.ie
- Noveno Andrew Martin, L00196906@atu.ie
- Vaibhav Krishnanand Revankar L00213629@atu.ie
- Ivan Kenneth Bibat, L00167791@atu.ie

# Setup
1. Download `train.csv` and `evaluation.csv` from the portal into `data/` (git-ignored).
2. Create the Python environment (experiments 02 onwards; 00 and 01 need only plain Python):
   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```

# Running an experiment
Every experiment from 03 onwards has two modes with the same procedure:

```bash
# 1. validation: cutoff day 5 (2025-01-25), prints AUC overall and per segment, writes nothing
.venv/bin/python experiments/05_click_aug/click_aug.py --cutoff 5 \
    --train data/train.csv --evaluation data/evaluation.csv

# 2. submission: official cutoff (day 7 = 2025-01-27), scores evaluation.csv
.venv/bin/python experiments/05_click_aug/click_aug.py --cutoff 7 \
    --train data/train.csv --evaluation data/evaluation.csv --out submissions/05_click_aug.csv

# 3. always check a file before uploading it
.venv/bin/python src/check_submission.py submissions/05_click_aug.csv
```

A full run takes about 10-15 minutes on a laptop. Features and MF scores are cached in
`data/cache/` (keyed by a hash of the code that produces them), so re-runs only redo what
changed. The cache can be deleted at any time; old entries are never reused.

## How we validate
Local validation follows the "History and time split" rules in `docs/phase_1_instructions.md`
exactly, so that local scores track the leaderboard:
- **Test events** = each user's *first* event at or after the cutoff (like `evaluation.csv`);
  later events of those users are not used at all.
- **No labels from the test events**: click-rate statistics are frozen at the cutoff (inside the
  fit period they update once per day). Click histories are inputs and are used as of each event.
- **MF scores** for fit day *d* come from an MF trained on days before *d* only.
- **Every setting** (MF regularisation, extra-loss weight, number of trees) is chosen inside the
  fit period (usually train on days 0-3, score day 4), never on the test events.

Validation AUC is lower than the leaderboard because the validation fit period has 5 days of
data instead of 7; differences between experiments have tracked the leaderboard so far.

# Working Tree
```text
techarena2026-atu/
├── data/                         # Local datasets (git-ignored, not in the repo)
│   ├── train.csv                 # Labelled training data (~7.3 GB)
│   ├── evaluation.csv            # Evaluation candidates to predict (no 'clicked' labels)
│   └── cache/                    # Cached features / MF scores written by the experiments
├── docs/
│   ├── phase_1_instructions.md   # Phase 1 task, data, and submission rules
│   └── example_submission.csv    # Reference submission format
├── experiments/
│   ├── RESULTS.md                # Scores of every experiment (validation + leaderboard)
│   ├── 00_starter/               # Provided baseline (keep unmodified)
│   ├── 01_history_recency/       # As-of features from the click histories (plain Python)
│   ├── 02_mf_lgbm/               # Matrix factorisation + LightGBM (old validation, superseded)
│   ├── 03_timesplit/             # The validation/submission pipeline used from here on
│   ├── 04_history_mf/            # 03 with a history-only MF (no per-user embedding)
│   ├── 05_click_aug/             # 04 + past clicks as extra MF training examples
│   ├── 06_mf_seeds/              # 05 with the MF averaged over 5 seeds
│   ├── 07_mf_seeds10/            # 06 with 10 seeds
│   ├── 08_choice_mf/             # 07 with a choice-model loss in the MF (one pick or none per event)
│   ├── 09_full_history/          # 07 with the user vector over all known clicks, not just the last 20
│   ├── 10_other_clicks/          # 09 with each click's other clicks as context for learning item vectors
│   └── 11_tied_items/            # 10 with one shared vector per item
├── src/
│   ├── metrics.py                # Global ROC AUC (same definition as the leaderboard)
│   └── check_submission.py       # Checks a submission against the Phase 1 rules
├── submissions/                  # One CSV per experiment
├── requirements.txt              # Python 3.12 environment
├── .gitignore
└── README.md
```

## How the experiments depend on each other
Later experiments reuse earlier code instead of copying it:
- `05_click_aug` and `04_history_mf` run `03_timesplit/timesplit.py` with their own MF stage.
- `03_timesplit` uses the feature builder of `01_history_recency` and the LightGBM helpers of
  `02_mf_lgbm`.

So **editing an older experiment changes the newer ones** (the cache keys make sure stale
results are not reused, but the scores can change). To try an idea, copy the latest experiment
into a new numbered folder rather than editing an old one.

The MF stage always runs in a separate process: PyTorch and LightGBM each bundle their own
OpenMP runtime on macOS and crash when loaded into the same process.
