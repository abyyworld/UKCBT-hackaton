"""Actor fingerprints inside an episode.

One actor running every wallet in a planted scenario tends to reuse settings: the same fee (normal look-alikes
use all three fee levels 97% of the time) and a wider spread of liquidity providers. These features measure how
consistent each transaction's episode is.
"""
import numpy as np
import pandas as pd

from src.features.episodes import _components

SCALES = [("all", 90), ("all", 300), ("all", 600), ("novel", 1800), ("novel", 3600), ("novel", 7200)]


def build(txs):
    txs = txs.reset_index(drop=True)
    n = len(txs)
    t = txs["ts"].to_numpy().astype(float)
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    fee = np.round(txs["fee_xrp"].to_numpy() * 1e6).astype(int)
    lp = txs["lp"].to_numpy()
    legs = txs["n_legs"].to_numpy()
    ain = txs["asset_in"].to_numpy()
    pair = pd.Series(list(zip(s, r)))
    pair_new = (pair.map(pair.value_counts()) == 1).to_numpy()
    idx = np.arange(n)
    f = pd.DataFrame(index=txs["tx"].to_numpy())
    for kind, T in SCALES:
        keep = np.ones(n, bool) if kind == "all" else pair_new
        ep = _components(np.r_[s[keep], r[keep]], np.r_[t[keep], t[keep]], np.r_[idx[keep], idx[keep]], n, T)
        if kind != "all":
            ep = np.where(keep, ep, -1 - idx)
        d = pd.DataFrame({"ep": ep, "fee": fee, "lp": np.where(lp == "", np.nan, lp), "legs": legs, "ain": ain})
        size = d.groupby("ep")["fee"].transform("size").to_numpy()
        same_fee = d.groupby(["ep", "fee"])["fee"].transform("size").to_numpy()
        name = f"{kind}{T}"
        f[f"{name}_same_fee_share"] = np.where(size > 1, (same_fee - 1) / np.maximum(size - 1, 1), np.nan)
        f[f"{name}_n_fees"] = d.groupby("ep")["fee"].transform("nunique").to_numpy()
        f[f"{name}_fee_mode_share"] = d.groupby("ep")["fee"].transform(lambda x: x.value_counts().iloc[0] / len(x)).to_numpy()
        n_conv = d.groupby("ep")["lp"].transform("count").to_numpy()
        n_lp = d.groupby("ep")["lp"].transform("nunique").to_numpy()
        f[f"{name}_lp_diversity"] = np.where(n_conv > 1, n_lp / np.maximum(n_conv, 1), np.nan)
        f[f"{name}_same_ain_share"] = d.groupby(["ep", "ain"])["ain"].transform("size").to_numpy() / size
        f[f"{name}_legs_mean"] = d.groupby("ep")["legs"].transform("mean").to_numpy()
    return f.astype(float)
