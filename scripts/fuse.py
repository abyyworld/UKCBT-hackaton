"""Power-mean rank fusion of existing member files (no retraining, runs in seconds).

Our final blend is a plain mean of percentile ranks, which weighs a disagreement at rank 30,000 as much as one at
rank 30. AP only cares about the top, so this fuses the members' distance from the top instead:

    d_i = 1 - pct_rank_i  (0 = the member's most suspicious row)
    score = 1 - (mean_i d_i ** p) ** (1 / p)

p = 1 is exactly the current rank blend. p < 1 is top-heavy: one confident member pulls a row up (recall of
medium/hard rows that only one member sees). p > 1 is consensus: one doubtful member pulls a row down (fewer normal
look-alikes in the top 1%, where we sit at 94% precision vs 96% for the leaders). p = 0 is the geometric mean.
RRF (reciprocal rank fusion, k = 60) is written as a fourth, standard top-heavy option.

    python scripts/fuse.py                     # writes submissions/fuse_p0.5.csv, fuse_p2.csv, fuse_p4.csv, fuse_rrf.csv
    python scripts/fuse.py --p 3 --files submissions/v17.csv,submissions/v28.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

MEMBERS = ["v17", "v28", "v17r", "v26", "scheme", "v13cf"]


def fuse(ranks, p, weights):
    d = np.clip(1.0 - ranks, 1.0 / ranks.shape[1], 1.0)
    w = weights[:, None] / weights.sum()
    if p == 0:
        return 1.0 - np.exp((w * np.log(d)).sum(axis=0))
    return 1.0 - ((w * d ** p).sum(axis=0)) ** (1.0 / p)


def rrf(ranks, weights, k=60):
    n = ranks.shape[1]
    pos = n - ranks * n + 1.0  # 1 = top row
    return (weights[:, None] / (k + pos)).sum(axis=0)


def save(path, hashes, raw, base):
    # scores strictly inside (0, 1), no ties: rank by fused score, break ties with the current final blend
    order = np.lexsort((base, raw))
    score = np.empty(len(raw))
    score[order] = (np.arange(len(raw)) + 1.0) / (len(raw) + 1.0)
    sub = pd.DataFrame({"TRANSACTION_HASH": hashes, "score": score})
    assert len(sub) == 40101 and sub["TRANSACTION_HASH"].is_unique and sub["score"].between(0, 1).all()
    assert sub["score"].nunique() == len(sub)
    sub.to_csv(path, index=False, float_format="%.8f")
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--p", default="0.5,2,4")
    args = ap.parse_args()
    found = sorted(glob.glob("submissions/*.csv") + glob.glob("*.csv") + glob.glob("data/**/*.csv", recursive=True))
    print("csv files found:", *found, sep="\n  ")
    if args.files:
        files = args.files.split(",")
    else:
        files = [f"submissions/{m}.csv" for m in MEMBERS if os.path.exists(f"submissions/{m}.csv")]
        print("using members:", files)
    if len(files) < 2:
        sys.exit("need at least 2 member files: pass --files a.csv,b.csv,... from the list above")
    weights = np.array([float(x) for x in args.weights.split(",")] if args.weights else [1.0] * len(files))
    subs = [pd.read_csv(f).set_index("TRANSACTION_HASH")["score"] for f in files]
    hashes = subs[0].index.to_numpy()
    ranks = np.vstack([s.loc[hashes].rank(pct=True).to_numpy() for s in subs])
    base = ranks.mean(axis=0)
    for f, r in zip(files, ranks):
        print(f"{f}: rank corr with mean blend {np.corrcoef(r, base)[0, 1]:.4f}")
    for p in [float(x) for x in args.p.split(",")]:
        raw = fuse(ranks, p, weights)
        top = np.argsort(-raw)[:402]
        print(f"p={p:g}: top 1% overlap with current blend {len(set(top) & set(np.argsort(-base)[:402]))}/402")
        save(f"submissions/fuse_p{p:g}.csv", hashes, raw, base)
    save("submissions/fuse_rrf.csv", hashes, rrf(ranks, weights), base)
