"""Lift slow-closure payments that sit just below the top 1%.

    python scripts/medium_boost.py --base submissions/blend12.csv --boost 0.01 --out mboost10

Payments ranked between the 97th and 99th percentile of the base ranking whose first-time-pair neighbourhood closes up
over 4 hours (src/features/slow.py: closure clo14400 >= 1 and density dens14400 >= 0.05) are 79% laundering on train
(out-of-fold v17 ranking), against 68.5% for that band overall, and it holds on both halves of the scenarios. Their
percentile is raised by --boost. On train out-of-fold v17 scores, a boost of 0.01 lifts AP from 67.47 to 69.18
(+1.0 and +1.8 on the two halves).
"""
import argparse
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.features import build_split  # noqa: E402
from src.predict import write  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--boost", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    test_txs, X = build_split("test", ["slow"])
    tx = test_txs["tx"].to_numpy()
    base = pd.read_csv(args.base).set_index("TRANSACTION_HASH")["score"].reindex(tx).rank(pct=True).to_numpy()
    flag = ((base >= 0.97) & (base < 0.99) & (X["slow__clo14400"].to_numpy() >= 1)
            & (X["slow__dens14400"].to_numpy() >= 0.05))
    score = base.copy()
    score[flag] += args.boost
    write(args.out, test_txs, score, base)
    print(f"boosted {int(flag.sum())} payments by {args.boost}")
