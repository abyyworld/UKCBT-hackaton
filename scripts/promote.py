"""Lift a second model's strongest rows into the base ranking's top 1%, leaving the rest of the base order alone.

    python scripts/promote.py --base submissions/blend6.csv --promoter blend_inputs/ltr.csv --n 40 --out blend6_ltr40

Takes the --n payments that the promoter ranks highest among those the base leaves outside its top 1%, and puts
them in the last --n places of the base top 1% (scripts/medium_quota.py has the same rule with out-of-fold checks).
With --top they go above everything instead.
"""
import argparse
import sys

import pandas as pd

sys.path.insert(0, ".")
from scripts.medium_quota import promote  # noqa: E402
from src.load import load_split  # noqa: E402
from src.predict import write  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--promoter", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", action="store_true", help="put the lifted rows above everything instead of at the bottom of the top 1%%")
    args = ap.parse_args()
    test_txs, _ = load_split("test")
    tx = test_txs["tx"].to_numpy()
    read = lambda p: pd.read_csv(p).set_index("TRANSACTION_HASH")["score"].reindex(tx).rank(pct=True).to_numpy()
    base = read(args.base)
    prom = read(args.promoter)
    score, pick = promote(base, prom, args.n)
    if args.top:
        score[pick] = base.max() + 1e-6 * (1 + prom[pick])
    write(args.out, test_txs, score, base)
    print(f"lifted {len(pick)} payments into the top 1%")
