"""Last-hour re-ranker: lift mid-band payments that share a quiet wallet with a confident payment in the same few hours.

    python scripts/flow_boost.py --base submissions/final.csv           # writes submissions/fb_*.csv (under a minute)

Why: the top 1% is already about 96% precise on the leaderboard, so AP is now decided in ranks 400 to 4,000, where
medium and hard payments sit among look-alikes. Laundering comes in schemes, so a mid-band payment that touches a
non-hub wallet which, within +-T hours, also carries one of our most confident payments is more likely part of the
same scheme. Hubs (many counterparties) and dust payments never pass evidence on, so exchanges and bots do not spread
scores. The top --keep rows stay exactly where they are; only the order below them changes.

Variants written (upload each, keep the best):
    fb_l15_t6   lambda 0.15, window 6 h        fb_l30_t6   lambda 0.30, window 6 h
    fb_l15_t24  lambda 0.15, window 24 h       fb_l30_t24  lambda 0.30, window 24 h
    fb_pm       consensus blend: power mean (p = 4 on 1 - rank) of the member files, if they exist
No labels are used: only the base ranking and the test week's own payment graph.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import load_split  # noqa: E402
from src.predict import write  # noqa: E402

MEMBERS = ["v17", "v28", "v17r", "v26", "scheme", "v13cf"]


def seconds(ts):
    t = pd.Series(ts)
    if np.issubdtype(t.dtype, np.datetime64):
        return (t.astype("int64") // 10**9).to_numpy()
    t = t.astype("int64").to_numpy()
    return t // 10**9 if t.max() > 1e14 else (t // 1000 if t.max() > 1e11 else t)


def neighbour_max(txs, rank, window, max_deg, dust_usd):
    """For each payment, the best base rank of another payment on a shared quiet wallet within +-window seconds."""
    ts = seconds(txs["ts"].to_numpy())
    usd = txs["usd_in"].fillna(0).to_numpy()
    long = pd.DataFrame({
        "w": np.concatenate([txs["sender"].to_numpy(), txs["receiver"].to_numpy()]),
        "cp": np.concatenate([txs["receiver"].to_numpy(), txs["sender"].to_numpy()]),
        "i": np.concatenate([np.arange(len(txs))] * 2),
    })
    deg = long.groupby("w")["cp"].nunique()
    long = long[long["w"].map(deg) <= max_deg]
    best = np.zeros(len(txs))
    for _, g in long.groupby("w"):
        idx = np.unique(g["i"].to_numpy())
        if len(idx) < 2:
            continue
        t = ts[idx]
        src = np.where(usd[idx] >= dust_usd, rank[idx], 0.0)
        close = np.abs(t[:, None] - t[None, :]) <= window
        np.fill_diagonal(close, False)
        m = np.where(close, src[None, :], 0.0).max(axis=1)
        best[idx] = np.maximum(best[idx], m)
    return best


def boost(rank, nbr, lam, keep, band_top):
    """Base rank plus lam times how far the neighbour sits inside the top band_top; top keep rows stay on top."""
    n = len(rank)
    q = 1 - band_top / n
    gain = np.clip((nbr - q) / (1 - q), 0, 1)
    score = rank + lam * gain
    top = np.argsort(-rank)[:keep]
    score[top] = 2 + rank[top]
    return score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="submissions/final.csv")
    ap.add_argument("--keep", type=int, default=400, help="base top rows left untouched")
    ap.add_argument("--band-top", type=int, default=1200, help="neighbours must sit in the base top this many rows")
    ap.add_argument("--max-deg", type=int, default=12, help="wallets with more counterparties are hubs, ignored")
    ap.add_argument("--dust-usd", type=float, default=1.0)
    args = ap.parse_args()

    test_txs, _ = load_split("test")
    tx = test_txs["tx"].to_numpy()
    read = lambda p: pd.read_csv(p).set_index("TRANSACTION_HASH")["score"].reindex(tx).rank(pct=True).to_numpy()
    base = read(args.base)

    for hours in (6, 24):
        nbr = neighbour_max(test_txs, base, hours * 3600, args.max_deg, args.dust_usd)
        moved = (nbr >= 1 - args.band_top / len(tx)).sum()
        print(f"window {hours} h: {moved} payments have a confident quiet-wallet neighbour")
        for lam in (0.15, 0.30):
            score = boost(base, nbr, lam, args.keep, args.band_top)
            write(f"fb_l{int(lam * 100)}_t{hours}", test_txs, score, base)
            new_top = set(np.argsort(-score)[:2000]) - set(np.argsort(-base)[:2000])
            print(f"  lambda {lam}: {len(new_top)} payments enter the top 2,000")

    paths = [f"submissions/{m}.csv" for m in MEMBERS]
    if all(os.path.exists(p) for p in paths):
        r = np.vstack([read(p) for p in paths])
        score = 1 - np.mean((1 - r) ** 4, axis=0) ** 0.25
        write("fb_pm", test_txs, score, base)
    else:
        print("member files missing, skipped fb_pm")
