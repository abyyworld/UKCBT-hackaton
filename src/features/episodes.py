"""Episode features: link transactions that share a wallet within T seconds, then describe each linked group.

Planted motifs are short contiguous episodes, so the episode a transaction belongs to says more than counts in
a fixed window. Several scales are used: tight ones for fast motifs, and coarse ones (linking only first-time
pairs) for the slower medium cases, so busy background wallets do not merge everything together.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SCALES = [("all", 30), ("all", 90), ("all", 180), ("all", 300), ("all", 600),
          ("novel", 1800), ("novel", 3600), ("novel", 7200), ("novel", 21600)]


def _components(wallets, times, tx_index, n, T):
    """Union transactions whose events on the same wallet are within T seconds of each other."""
    order = np.lexsort((times, wallets))
    w, t, i = wallets[order], times[order], tx_index[order]
    same = (w[1:] == w[:-1]) & ((t[1:] - t[:-1]) <= T)
    a, b = i[:-1][same], i[1:][same]
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _episode_stats(ep, txs, pair_new):
    d = pd.DataFrame({
        "ep": ep, "ts": txs["ts"].to_numpy().astype(float), "s": txs["sender"].to_numpy(),
        "r": txs["receiver"].to_numpy(), "lu": np.log1p(txs["usd_in"].to_numpy()),
        "conv": txs["is_conversion"].to_numpy().astype(float),
        "xrp_out": (txs["asset_out"].to_numpy() == "XRP").astype(float),
        "aout": txs["asset_out"].to_numpy(), "new": pair_new.astype(float),
    })
    d = d.sort_values(["ep", "ts"])
    d["gap"] = d.groupby("ep")["ts"].diff()
    d["dlu"] = d.groupby("ep")["lu"].diff()
    g = d.groupby("ep")
    st = pd.DataFrame({
        "size": g.size(),
        "span": np.log1p(g["ts"].max() - g["ts"].min()),
        "gap_med": np.log1p(g["gap"].median()),
        "gap_cv": g["gap"].std() / (g["gap"].mean() + 1),
        "usd_med": g["lu"].median(),
        "usd_std": g["lu"].std(),
        "usd_range": g["lu"].max() - g["lu"].min(),
        "drift_down": (g["dlu"].apply(lambda x: (x < 0).mean() if x.notna().any() else np.nan)),
        "drift_abs": g["dlu"].apply(lambda x: x.abs().mean()),
        "conv": g["conv"].mean(),
        "xrp_out": g["xrp_out"].mean(),
        "n_assets_out": g["aout"].nunique(),
        "new_share": g["new"].mean(),
        "n_send": g["s"].nunique(),
        "n_recv": g["r"].nunique(),
    })
    wallets = pd.concat([d[["ep", "s"]].rename(columns={"s": "w"}), d[["ep", "r"]].rename(columns={"r": "w"})])
    st["n_wallets"] = wallets.groupby("ep")["w"].nunique()
    both = d.groupby("ep").apply(lambda x: len(set(x["s"]) & set(x["r"])), include_groups=False)
    st["relay_share"] = both / st["n_wallets"]
    pairs = d.drop_duplicates(["ep", "s", "r"]).groupby("ep").size()
    st["cyclomatic"] = pairs - st["n_wallets"] + 1
    st["max_out_share"] = d.groupby(["ep", "s"]).size().groupby("ep").max() / st["size"]
    st["max_in_share"] = d.groupby(["ep", "r"]).size().groupby("ep").max() / st["size"]
    st["density"] = pairs / (st["n_send"] * st["n_recv"])
    # where this transaction sits inside its episode
    pos = d.groupby("ep").cumcount() / d["ep"].map(st["size"]).clip(lower=1)
    rel = d["lu"] - d["ep"].map(st["usd_med"])
    return st, pos.reindex(range(len(txs))).to_numpy(), rel.reindex(range(len(txs))).to_numpy()


def build(txs):
    txs = txs.reset_index(drop=True)
    n = len(txs)
    t = txs["ts"].to_numpy().astype(float)
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    pair = pd.Series(list(zip(s, r)))
    pair_new = (pair.map(pair.value_counts()) == 1).to_numpy()
    idx = np.arange(n)
    f = pd.DataFrame(index=txs["tx"].to_numpy())
    for kind, T in SCALES:
        keep = np.ones(n, bool) if kind == "all" else pair_new
        wallets = np.r_[s[keep], r[keep]]
        times = np.r_[t[keep], t[keep]]
        tx_index = np.r_[idx[keep], idx[keep]]
        ep = _components(wallets, times, tx_index, n, T)
        if kind != "all":
            ep = np.where(keep, ep, -1 - idx)  # transactions outside the linkable set stay alone
        st, pos, rel = _episode_stats(ep, txs, pair_new)
        name = f"{kind}{T}"
        for c in st.columns:
            f[f"{name}_{c}"] = pd.Series(ep).map(st[c]).to_numpy()
        f[f"{name}_pos"] = pos
        f[f"{name}_rel_usd"] = rel
    return f.astype(float)
