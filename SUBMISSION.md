# How to reproduce our predictions

Team **Batman and Robin**, Decode & Detect, 24 October 2026.

## 1. Requirements

- Python 3.13
- Measured: 7.6 minutes from raw data to the upload file on a 4-core Linux machine with 15 GB of RAM (feature caches
  empty). No GPU and no internet needed once the packages are installed.

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
It runs three steps (see `run.py`): stage 1 LightGBM on the `local` features, stage 2 collective scoring on
`local` and `episodes`, then a rank average of the two. Each model is averaged over 3 seeds.
Feature caches go to `data/cache/`; delete that folder to rebuild from scratch.

All randomness uses fixed seeds set in the code, so the same data gives the same file.

To measure our validation score (grouped 5-fold out-of-fold average precision on `train/`):

```sh
.venv/bin/python -m src.cv
```

## 4. Results

- Out-of-fold AP (grouped 5-fold): TODO
- Leaderboard score of the submitted file: TODO
- Commit: TODO
