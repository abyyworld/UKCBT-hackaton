"""Reserve part of the top 1% for medium-looking payments the main ranking misses.

    python scripts/medium_quota.py --base submissions/v17_realmotif.csv --n 40 --out v17_mq40

The leaderboard reviews the top 1% of each ranking. Medium laundering (the same motifs stretched over hours) sits
just below that band in our main models. This takes the payments that the medium expert of
scripts/family_experts.py ranks highest but the base ranking leaves outside its top 1%, and lifts the --n strongest
of them into the last --n places of the top 1% (the base's best rows keep their order above them).
On train the same rule runs on out-of-fold scores, to show how many medium cases it brings into the top 1% and
what it does to average precision. --n is per 40,000 payments.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from src.cv import training_frame  # noqa: E402
from src.load import load_split  # noqa: E402
from src.predict import write  # noqa: E402


def promote(base, med, n, band=0.01):
    """base, med: percentile ranks. Lift the n rows with the highest med outside the base top band."""
    base = base.copy()
    k = int(round(band * len(base)))
    order = np.sort(base)[::-1]
    outside = np.where(base < order[k - 1])[0]
    pick = outside[np.argsort(-med[outside])[:n]]
    # the lifted rows take the last n places of the top band, just below the base's (k - n) best rows
    anchor = order[max(k - n - 1, 0)]
    base[pick] = anchor - 1e-9 * (2 - med[pick])
    return base, pick


def medium_in_top(score, lab, band=0.01):
    top = pd.Series(score, index=lab.index).rank(ascending=False) <= round(band * len(lab))
    med = (lab["difficulty"] == "medium") & (lab["label"] == 1)
    return int((top & med).sum()), float(lab["label"][top].mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submissions/v17_realmotif.csv")
    ap.add_argument("--base_oof", default="data/oof/v17_realmotif.parquet")
    ap.add_argument("--experts", default="data/oof/fam_experts.parquet")
    ap.add_argument("--experts_test", default="data/oof/fam_experts_test.parquet")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    _, lab, _ = training_frame(["local"])
    y = lab["label"].to_numpy()
    b = pd.read_parquet(args.base_oof).set_index("tx")["oof"].reindex(lab.index).rank(pct=True).to_numpy()
    m = pd.read_parquet(args.experts).set_index("tx")["medium"].reindex(lab.index).rank(pct=True).to_numpy()
    print(f"train OOF, base: AP {100 * average_precision_score(y, b):.2f}, medium in top 1%% %d, precision %.2f"
          % medium_in_top(b, lab))
    for n in (10, 20, 40, 60, 80, 120):
        s, pick = promote(b, m, n)
        print(f"  n={n:4d}: AP {100 * average_precision_score(y, s):.2f}, lifted {len(pick)} "
              f"(laundering {int(y[pick].sum())}, medium {int(((lab['difficulty'] == 'medium') & (lab['label'] == 1)).to_numpy()[pick].sum())}),"
              f" medium in top 1%% %d, precision %.2f" % medium_in_top(s, lab))
    if args.out:
        test_txs, _ = load_split("test")
        tx = test_txs["tx"].to_numpy()
        bt = pd.read_csv(args.base).set_index("TRANSACTION_HASH")["score"].reindex(tx).rank(pct=True).to_numpy()
        mt = pd.read_parquet(args.experts_test).set_index("tx")["medium"].reindex(tx).rank(pct=True).to_numpy()
        n = int(round(args.n * len(tx) / len(lab)))
        s, pick = promote(bt, mt, n)
        write(args.out, test_txs, s, bt)
        print(f"test: lifted {len(pick)} payments into the top 1%")
