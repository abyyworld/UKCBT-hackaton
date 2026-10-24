"""Scheme isolation at the episode level.

One actor controls the wallets of a planted scheme, so during the scheme those wallets make no other payments.
Normal look-alike wallets keep their recurring relationships going. For each episode (see episodes.py) count
the recurring-pair payments its wallets make during the episode and in the half hour around it.
"""
import numpy as np
import pandas as pd

from src.features.episodes import _components

SCALES = [("all", 90), ("all", 300), ("all", 600), ("novel", 1800), ("novel", 3600)]
PAD = 1800


def _events(wallets, times):
    order = np.lexsort((times, wallets))
    w, t = wallets[order], times[order]
    starts = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1]
    return {w[s]: t[s:e] for s, e in zip(starts, np.r_[starts[1:], len(w)])}


def build(txs):
    txs = txs.reset_index(drop=True)
    n = len(txs)
    t = txs["ts"].to_numpy().astype(float)
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    pair = pd.Series(list(zip(s, r)))
    pair_n = pair.map(pair.value_counts()).to_numpy()
    new, rec = pair_n == 1, pair_n >= 2
    rec_ev = _events(np.r_[s[rec], r[rec]], np.r_[t[rec], t[rec]])
    any_ev = _events(np.r_[s, r], np.r_[t, t])
    idx = np.arange(n)
    f = pd.DataFrame(index=txs["tx"].to_numpy())
    for kind, T in SCALES:
        keep = np.ones(n, bool) if kind == "all" else new
        ep = _components(np.r_[s[keep], r[keep]], np.r_[t[keep], t[keep]], np.r_[idx[keep], idx[keep]], n, T)
        ep = np.where(keep, ep, -1 - idx)
        d = pd.DataFrame({"ep": ep, "t": t, "s": s, "r": r, "rec": rec})
        g = d.groupby("ep")
        size = g.size()
        multi = size[size >= 2].index
        t0, t1 = g["t"].min(), g["t"].max()
        own_rec = g["rec"].sum()
        during, around, any_around = {}, {}, {}
        members = d[d["ep"].isin(multi)]
        wallets = pd.concat([members[["ep", "s"]].rename(columns={"s": "w"}),
                             members[["ep", "r"]].rename(columns={"r": "w"})]).drop_duplicates()
        for e, grp in wallets.groupby("ep"):
            a, b = t0[e], t1[e]
            c_in = c_pad = c_any = 0
            for w in grp["w"]:
                ev = rec_ev.get(w)
                if ev is not None:
                    c_in += np.searchsorted(ev, b, "right") - np.searchsorted(ev, a, "left")
                    c_pad += np.searchsorted(ev, b + PAD, "right") - np.searchsorted(ev, a - PAD, "left")
                av = any_ev[w]
                c_any += np.searchsorted(av, b + PAD, "right") - np.searchsorted(av, a - PAD, "left")
            nw = len(grp)
            # payments inside the episode that are themselves recurring pairs are counted twice (once per wallet)
            during[e] = (c_in - 2 * own_rec[e]) / nw
            around[e] = (c_pad - 2 * own_rec[e]) / nw
            any_around[e] = (c_any - 2 * size[e]) / nw
        name = f"{kind}{T}"
        f[f"{name}_rec_during"] = pd.Series(ep).map(during).to_numpy()
        f[f"{name}_rec_around"] = pd.Series(ep).map(around).to_numpy()
        f[f"{name}_other_around"] = pd.Series(ep).map(any_around).to_numpy()
        f[f"{name}_rec_rate"] = f[f"{name}_rec_during"] / (np.log1p(pd.Series(ep).map(t1 - t0).to_numpy()) + 1)
    return f.astype(float)
