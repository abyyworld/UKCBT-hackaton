"""Rank-average several prediction files into one upload.

    python -m src.blend --out v3_blend submissions/v1_local.csv submissions/v2_stage2.csv [--weights 1,1]
"""
import argparse

import numpy as np
import pandas as pd

from src.load import load_split
from src.predict import write


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="")
    args = ap.parse_args()
    weights = [float(w) for w in args.weights.split(",")] if args.weights else [1.0] * len(args.files)
    test_txs, _ = load_split("test")
    ranks = []
    for path in args.files:
        sub = pd.read_csv(path).set_index("TRANSACTION_HASH").loc[test_txs["tx"].to_numpy()]
        ranks.append(sub["score"].rank(pct=True).to_numpy())
    raw = np.average(np.vstack(ranks), axis=0, weights=weights)
    write(args.out, test_txs, raw, ranks[0])
