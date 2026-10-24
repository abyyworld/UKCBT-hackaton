"""Motif shape features: separate fast planted motifs from fast normal look-alikes.

Link first-time-pair transactions that share a wallet within T seconds (as in episodes.py), then describe the
flow through each linked group: how much value intermediate wallets keep, how regular the hop delays are, how
amounts change hop to hop, how many wallets only pass money through, and whether the wallets do anything else
around the group in time (planted wallets tend to be quiet outside the scheme).
"""
import numpy as np
import pandas as pd

from src.features.episodes import _components

SCALES = [60, 300, 900]
QUIET = 6 * 3600


def _wallet_outside(txs, ep, t):
    """Per (group, wallet): transactions of the wallet within +-6 h of the group span that are not in the group."""
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    ev = pd.DataFrame({"w": np.r_[s, r], "ts": np.r_[t, t], "ep": np.r_[ep, ep]})
    ev = ev.sort_values(["w", "ts"]).reset_index(drop=True)
    span = ev.groupby("ep")["ts"].agg(["min", "max"])
    gw = ev.drop_duplicates(["ep", "w"])[["ep", "w"]].copy()
    gw = gw.join(span, on="ep")
    out = np.zeros(len(gw))
    total = np.zeros(len(gw))
    inside = ev.groupby(["ep", "w"]).size()
    # count all events of the wallet in [min-QUIET, max+QUIET] with searchsorted per wallet
    starts = ev.groupby("w").indices
    wts = ev["ts"].to_numpy()
    for k, (w, lo, hi) in enumerate(zip(gw["w"].to_numpy(), gw["min"].to_numpy(), gw["max"].to_numpy())):
        idx = starts[w]
        a, b = idx[0], idx[-1] + 1
        seg = wts[a:b]
        total[k] = np.searchsorted(seg, hi + QUIET, "right") - np.searchsorted(seg, lo - QUIET, "left")
    gw["inside"] = inside.loc[list(zip(gw["ep"], gw["w"]))].to_numpy()
    gw["outside"] = total - gw["inside"].to_numpy()
    return gw


def _group_stats(ep, txs, t, wall_med):
    d = pd.DataFrame({
        "ep": ep, "ts": t, "s": txs["sender"].to_numpy(), "r": txs["receiver"].to_numpy(),
        "ui": txs["usd_in"].to_numpy(), "uo": txs["usd_out"].to_numpy(),
        "ai": txs["asset_in"].to_numpy(), "ao": txs["asset_out"].to_numpy(),
    })
    size = d.groupby("ep").size()
    multi = size[size > 1].index
    d = d[d["ep"].isin(multi)].sort_values(["ep", "ts"])
    g = d.groupby("ep")
    st = pd.DataFrame(index=multi)
    gap = g["ts"].diff()
    d["gap"] = gap
    st["size"] = g.size()
    st["span"] = np.log1p(g["ts"].max() - g["ts"].min())
    st["gap_med"] = np.log1p(d.groupby("ep")["gap"].median())
    st["gap_cv"] = d.groupby("ep")["gap"].std() / (d.groupby("ep")["gap"].mean() + 1)
    st["gap_maxmin"] = np.log1p(d.groupby("ep")["gap"].max()) - np.log1p(d.groupby("ep")["gap"].min())
    lu = np.log1p(d["ui"])
    d["dlu"] = lu.groupby(d["ep"]).diff()
    st["hop_ratio_med"] = d.groupby("ep")["dlu"].median()
    st["hop_ratio_abs"] = d["dlu"].abs().groupby(d["ep"]).mean()

    inn = d.groupby(["ep", "r"]).agg(in_usd=("uo", "sum"), in_first=("ts", "min"), in_last=("ts", "max"),
                                     in_n=("ts", "size"), in_asset=("ao", "first"))
    out = d.groupby(["ep", "s"]).agg(out_usd=("ui", "sum"), out_first=("ts", "min"), out_last=("ts", "max"),
                                     out_n=("ts", "size"), out_asset=("ai", "first"))
    inn.index.names = out.index.names = ["ep", "w"]
    w = inn.join(out, how="outer")
    w["mid"] = w["in_n"].notna() & w["out_n"].notna()
    w["src"] = w["in_n"].isna()
    w["sink"] = w["out_n"].isna()
    mid = w[w["mid"]].copy()
    mid["kept"] = np.log(mid["out_usd"] + 1) - np.log(mid["in_usd"] + 1)
    mid["delay"] = np.log1p((mid["out_first"] - mid["in_first"]).clip(lower=0))
    mid["after"] = (mid["out_first"] >= mid["in_first"]).astype(float)
    mid["conv"] = (mid["in_asset"] != mid["out_asset"]).astype(float)
    gm = mid.groupby(level="ep")
    st["n_mid"] = gm.size()
    st["kept_med"] = gm["kept"].median()
    st["kept_absmax"] = gm["kept"].apply(lambda x: x.abs().max())
    st["kept_std"] = gm["kept"].std()
    st["delay_med"] = gm["delay"].median()
    st["delay_std"] = gm["delay"].std()
    st["after_share"] = gm["after"].mean()
    st["mid_conv"] = gm["conv"].mean()
    gw = w.groupby(level="ep")
    nw = gw.size()
    st["mid_share"] = st["n_mid"].reindex(multi).fillna(0) / nw
    st["n_mid"] = st["n_mid"].fillna(0)
    st["src_share"] = gw["src"].mean()
    st["sink_share"] = gw["sink"].mean()
    st["max_in"] = gw["in_n"].max()
    st["max_out"] = gw["out_n"].max()
    st["n_src"] = gw["src"].sum()
    st["n_sink"] = gw["sink"].sum()
    # end-to-end conservation: value leaving sinks vs value entering from sources
    src_usd = w[w["src"]].groupby(level="ep")["out_usd"].sum()
    sink_usd = w[w["sink"]].groupby(level="ep")["in_usd"].sum()
    st["e2e_kept"] = np.log1p(sink_usd) - np.log1p(src_usd)
    st["n_assets"] = pd.concat([d[["ep", "ai"]].rename(columns={"ai": "a"}),
                                d[["ep", "ao"]].rename(columns={"ao": "a"})]).groupby("ep")["a"].nunique()

    # group usd against the wallets' usual usd
    d["rel_s"] = lu - np.log1p(d["s"].map(wall_med))
    d["rel_r"] = lu - np.log1p(d["r"].map(wall_med))
    st["rel_usual"] = d.groupby("ep")[["rel_s", "rel_r"]].mean().mean(axis=1)
    return st


def build(txs):
    txs = txs.reset_index(drop=True)
    n = len(txs)
    t = txs["ts"].to_numpy().astype(float)
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    pair = pd.Series(list(zip(s, r)))
    pair_new = (pair.map(pair.value_counts()) == 1).to_numpy()
    lu = txs["usd_in"].to_numpy()
    wall_med = pd.concat([pd.Series(lu, index=s), pd.Series(lu, index=r)]).groupby(level=0).median()
    idx = np.arange(n)
    f = pd.DataFrame(index=txs["tx"].to_numpy())
    for T in SCALES:
        keep = pair_new
        ep = _components(np.r_[s[keep], r[keep]], np.r_[t[keep], t[keep]], np.r_[idx[keep], idx[keep]], n, T)
        ep = np.where(keep, ep, -1 - idx)
        st = _group_stats(ep, txs, t, wall_med)
        # quiet wallets: activity outside the group within +-6 h
        in_multi = pd.Series(ep).isin(st.index).to_numpy()
        gw = _wallet_outside(txs, ep, t)
        gw = gw[gw["ep"].isin(st.index)]
        go = gw.groupby("ep")
        st["out_mean"] = np.log1p(go["outside"].mean())
        st["quiet_share"] = go["outside"].apply(lambda x: (x == 0).mean())
        name = f"n{T}"
        epser = pd.Series(ep)
        for c in st.columns:
            f[f"{name}_{c}"] = epser.map(st[c]).to_numpy()
        # this transaction: did the receiver pass the money on inside the group, how fast, how much kept
        d = pd.DataFrame({"ep": ep, "ts": t, "s": s, "r": r, "u": txs["usd_out"].to_numpy(),
                          "ui": txs["usd_in"].to_numpy()})
        d = d[in_multi]
        nxt = d.sort_values("ts").groupby(["ep", "s"]).agg(nts=("ts", "min"), nu=("ui", "sum"))
        nxt.index.names = ["ep", "r"]
        j = d.join(nxt, on=["ep", "r"])
        fwd = (j["nts"] - j["ts"])
        col = np.full(n, np.nan)
        col[j.index] = np.log1p(fwd.clip(lower=0)).to_numpy()
        f[f"{name}_tx_fwd_delay"] = col
        col = np.full(n, np.nan)
        col[j.index] = (np.log1p(j["nu"]) - np.log1p(j["u"])).to_numpy()
        f[f"{name}_tx_fwd_kept"] = col
        outs = gw.set_index(["ep", "w"])["outside"]
        so = pd.Series(list(zip(d["ep"], d["s"]))).map(outs).to_numpy()
        ro = pd.Series(list(zip(d["ep"], d["r"]))).map(outs).to_numpy()
        col = np.full(n, np.nan); col[d.index] = np.log1p(so); f[f"{name}_tx_s_outside"] = col
        col = np.full(n, np.nan); col[d.index] = np.log1p(ro); f[f"{name}_tx_r_outside"] = col
    return f.astype(float)
