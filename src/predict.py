"""Fit on all of train, score the test split, write and check the upload file.

    python -m src.predict --tag v1 [--modules local] [--seeds 3]
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import src.cv as cv
from src.cv import PARAMS, training_frame
from src.features import MODULES, build_split

N_TEST = 40101


def fit_predict(X, y, X_test, seeds=3, params=None):
    pred = np.zeros(len(X_test))
    for seed in range(seeds):
        model = lgb.LGBMClassifier(**{**PARAMS, **(params or {}), "random_state": seed})
        sw = cv.SAMPLE_WEIGHT if cv.SAMPLE_WEIGHT is not None and len(cv.SAMPLE_WEIGHT) == len(y) else None
        model.fit(X, y, sample_weight=sw)
        pred += model.predict_proba(X_test[X.columns])[:, 1]
    return pred / seeds


def to_scores(raw, tie_break):
    """Rank-normalise into [0, 1] with a tiny continuous tie-breaker so no two scores tie."""
    order = pd.Series(raw + 1e-9 * tie_break).rank(method="first").to_numpy()
    return (order - 1) / (len(order) - 1)


def check(sub, test_hashes):
    assert list(sub.columns[:2]) == ["TRANSACTION_HASH", "score"], "wrong columns"
    assert len(sub) == N_TEST == len(test_hashes), f"expected {N_TEST} rows, got {len(sub)}"
    assert sub["TRANSACTION_HASH"].is_unique, "duplicate hashes"
    assert set(sub["TRANSACTION_HASH"]) == set(test_hashes), "hashes do not match the test data"
    assert np.isfinite(sub["score"]).all() and sub["score"].between(0, 1).all(), "scores outside [0, 1]"
    assert sub["score"].nunique() == len(sub), "tied scores"


def write(tag, test_txs, raw, tie_break):
    sub = pd.DataFrame({"TRANSACTION_HASH": test_txs["tx"].to_numpy(), "score": to_scores(raw, tie_break)})
    check(sub, test_txs["tx"].to_numpy())
    os.makedirs("submissions", exist_ok=True)
    path = f"submissions/{tag}.csv"
    sub.to_csv(path, index=False, float_format="%.8f")
    print(f"wrote {path}: {len(sub)} rows, checks passed")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    modules = args.modules.split(",")
    X, lab, _ = training_frame(modules)
    test_txs, X_test = build_split("test", modules)
    raw = fit_predict(X, lab["label"].to_numpy(), X_test, args.seeds)
    write(args.tag, test_txs, raw, X_test.iloc[:, 0].rank().to_numpy())
