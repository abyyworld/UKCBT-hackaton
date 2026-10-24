"""Add a plain-English reason column to a predictions file, leaving the scores unchanged.

    python scripts/add_reasons.py submissions/final.csv      # rewrites the file as TRANSACTION_HASH,score,reason

The reason describes the group of first-time counterparty payments the transaction belongs to (payments that share
a wallet within 30 minutes, from src/features/episodes.py): its size, how many wallets, how long it took and its
shape. Only the top 5% of scores get a pattern description; the rest are marked as low priority.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.features import build_split  # noqa: E402
from src.predict import check  # noqa: E402

E = "episodes__novel1800_"


def describe(row):
    size, wallets = int(row[E + "size"]), int(row[E + "n_wallets"])
    secs = float(np.expm1(row[E + "span"]))  # span is stored as log1p(seconds)
    took = "under a minute" if secs < 60 else f"{secs / 60:.0f} min"
    if size >= 3:
        if row[E + "max_out_share"] >= 0.5:
            shape = "one wallet sprays funds to new counterparties"
        elif row[E + "max_in_share"] >= 0.5:
            shape = "one wallet gathers funds from new counterparties"
        elif row[E + "cyclomatic"] >= 1:
            shape = "new wallets pay each other in loops or dense blocks"
        else:
            shape = "funds are layered through a chain of new wallets"
        if row[E + "conv"] >= 0.5:
            shape += " with currency conversions along the way"
        return f"{size} linked first-time payments among {wallets} wallets in {took}: {shape}"
    if row["local__pair_is_new"] >= 1:
        return "first-time counterparty pair inside a busy burst of activity"
    return "unusual timing and flow around a recurring counterparty pair"


if __name__ == "__main__":
    path = sys.argv[1]
    sub = pd.read_csv(path)[["TRANSACTION_HASH", "score"]]
    txs, X = build_split("test", ["local", "episodes"])
    feats = X.assign(tx=txs["tx"].to_numpy()).set_index("tx").reindex(sub["TRANSACTION_HASH"])
    top = sub["score"].to_numpy() >= np.quantile(sub["score"], 0.95)
    reasons = np.full(len(sub), "low priority: no strong laundering pattern", dtype=object)
    reasons[top] = [describe(r) for _, r in feats[top].iterrows()]
    sub["reason"] = reasons
    check(sub, txs["tx"].to_numpy())
    sub.to_csv(path, index=False)
    print(f"wrote {path} with reasons: {int(top.sum())} described, {len(sub) - int(top.sum())} low priority")
