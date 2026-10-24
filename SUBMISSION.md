# How to reproduce our predictions

Team **Batman and Robin**, Decode & Detect, 24 October 2026.

## 1. Requirements

- Python 3.13
- About 45 minutes from raw data to the upload file on a 4-core Linux machine with 15 GB of RAM (feature caches
  empty). TODO: measured time of the final check. No GPU and no internet needed once the packages are installed.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip install -r requirements.txt
```

## 2. Data

The data is not in this ZIP. Download both packages from <https://hackathon.ukcbt.org/> and unzip them so these
paths exist:

```
data/decode-and-detect-training-data/train/transactions.parquet
data/decode-and-detect-training-data/train/labels.csv
data/decode-and-detect-training-data/validation/transactions.parquet
data/decode-and-detect-test-data/transactions.parquet
```

## 3. Run

```sh
.venv/bin/python run.py
```

This writes `submissions/final.csv`: one row per test transaction (40,101 rows), columns `TRANSACTION_HASH,score`.
It builds six models and rank-averages them with `src/blend.py` (see the docstring of `run.py`):

| Member | What it is | Script |
|---|---|---|
| v17 | stage 1 LightGBM, a real-vs-look-alike model on motif-shaped payments, stage 2 group scoring | `scripts/realmotif_run.py` |
| v28 | the same recipe on within-week percentile features plus the `slow` closure features | `scripts/robust2_run.py` (`SPLIT_RANK=1`) |
| v17r | v17 without the amount-evenness features | `explore/robust_run.py` |
| v26 | rank average of v17 and a pseudo-labelling model cross-fitted on the unlabelled test week | `scripts/pseudo_adapt.py` |
| scheme | scores whole candidate schemes and passes the score to their payments | `scripts/scheme_model.py` |
| v13cf | stage 1 plus a fold-consistent stage 2 with first-time-pair neighbour scores | `src/stage2.py --consistent --extra ftp` |

Features: `local, episodes, isolation, patterns, shape` (and `slow` for v28). Each model is averaged over 2 seeds.
`python run.py --single` builds only v17 (about 10 minutes).
Feature caches go to `data/cache/`; delete that folder to rebuild from scratch.

All randomness uses fixed seeds set in the code, so the same data gives the same file.

To measure our validation score (grouped 5-fold out-of-fold average precision on `train/`):

```sh
.venv/bin/python -m src.cv
```

## 4. Results

- Out-of-fold AP (grouped 5-fold) of the members: v17 67.47 (stage 2 slightly optimistic, see `NOW.md`), v13cf
  61.96 with cohort-grouped folds (honest stage 2), scheme 60.39.
- Leaderboard score of the submitted file: **70.3** (rank blend of v17, v28, v17r, v26, scheme and v13cf).
  Single members on the leaderboard: v17 69.2, v17r 69.0, v28 68.8, v26 68.8.
- Note: v26 was built from an earlier pseudo-labelling run, so a fresh `run.py` gives a file that is very close to,
  but not byte-identical with, the uploaded one (rank correlation of v26 with the documented recipe 0.99).
- Commit: `9dce39e`
