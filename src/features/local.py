"""Local tempo, novelty and amount features for every transaction.

All windows are computed inside one split's own week. Nothing here reads labels.
"""
import numpy as np
import pandas as pd

WINDOWS = [120, 300, 900, 3600, 21600, 86400]
NOVEL_WINDOWS = [1800, 7200, 21600, 86400]
FAR = 7 * 86400.0


def _events(wallets, times):
    """Sorted event times per wallet."""
    order = np.lexsort((times, wallets))
    w, t = wallets[order], times[order]
    starts = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1]
    return {w[s]: t[s:e] for s, e in zip(starts, np.r_[starts[1:], len(w)])}


def _window_counts(events, wallets, times, before, after):
    """For each query (wallet, time) count that wallet's events in [time - before, time + after]."""
    out = np.zeros(len(wallets))
    df = pd.DataFrame({"w": wallets, "t": times, "i": np.arange(len(wallets))})
    for w, g in df.groupby("w", sort=False):
        ev = events.get(w)
        if ev is None:
            continue
        q = g["t"].to_numpy()
        out[g["i"].to_numpy()] = np.searchsorted(ev, q + after, "right") - np.searchsorted(ev, q - before, "left")
    return out


def _gap(events, wallets, times, direction):
    """Seconds to the wallet's nearest event strictly after (direction=+1) or before (-1) the query time."""
    out = np.full(len(wallets), FAR)
    df = pd.DataFrame({"w": wallets, "t": times, "i": np.arange(len(wallets))})
    for w, g in df.groupby("w", sort=False):
        ev = events.get(w)
        if ev is None or len(ev) == 0:
            continue
        q = g["t"].to_numpy()
        if direction > 0:
            j = np.searchsorted(ev, q, "right")
            ok = j < len(ev)
            out[g["i"].to_numpy()[ok]] = ev[j[ok]] - q[ok]
        else:
            j = np.searchsorted(ev, q, "left") - 1
            ok = j >= 0
            out[g["i"].to_numpy()[ok]] = q[ok] - ev[j[ok]]
    return out


def _window_sum(events_t, events_v, wallets, times, before, after):
    out = np.zeros(len(wallets))
    df = pd.DataFrame({"w": wallets, "t": times, "i": np.arange(len(wallets))})
    for w, g in df.groupby("w", sort=False):
        ev = events_t.get(w)
        if ev is None:
            continue
        cs = np.r_[0.0, np.cumsum(events_v[w])]
        q = g["t"].to_numpy()
        hi = np.searchsorted(ev, q + after, "right")
        lo = np.searchsorted(ev, q - before, "left")
        out[g["i"].to_numpy()] = cs[hi] - cs[lo]
    return out


def _sorted_values(wallets, times, values):
    order = np.lexsort((times, wallets))
    w, v = wallets[order], values[order]
    starts = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1]
    return {w[s]: v[s:e] for s, e in zip(starts, np.r_[starts[1:], len(w)])}


def build(txs):
    t = txs["ts"].to_numpy().astype(float)
    s = txs["sender"].to_numpy()
    r = txs["receiver"].to_numpy()
    usd = txs["usd_in"].to_numpy()
    f = pd.DataFrame(index=txs["tx"].to_numpy())

    out_ev = _events(s, t)          # each wallet's sends
    in_ev = _events(r, t)           # each wallet's receives
    any_ev = _events(np.r_[s, r], np.r_[t, t])

    roles = {"s_out": (out_ev, s), "s_in": (in_ev, s), "r_in": (in_ev, r), "r_out": (out_ev, r)}
    for W in WINDOWS:
        for name, (ev, who) in roles.items():
            c = _window_counts(ev, who, t, W, W)
            if name in ("s_out", "r_in"):
                c -= 1  # do not count this transaction itself
            f[f"{name}_{W}"] = c
        f[f"both_any_{W}"] = _window_counts(any_ev, s, t, W, W) + _window_counts(any_ev, r, t, W, W) - 2
        f[f"s_in_before_{W}"] = _window_counts(in_ev, s, t, W, 0)
        f[f"r_out_after_{W}"] = _window_counts(out_ev, r, t, 0, W)

    f["gap_s_last_in"] = _gap(in_ev, s, t, -1)
    f["gap_r_next_out"] = _gap(out_ev, r, t, +1)
    f["gap_s_next_out"] = _gap(out_ev, s, t, +1)
    f["gap_s_prev_out"] = _gap(out_ev, s, t, -1)
    f["gap_r_prev_in"] = _gap(in_ev, r, t, -1)
    f["gap_r_next_in"] = _gap(in_ev, r, t, +1)
    for c in [c for c in f.columns if c.startswith("gap_")]:
        f[c] = np.log1p(f[c])
    f["gap_min_any"] = f[[c for c in f.columns if c.startswith("gap_")]].min(axis=1)

    # value flow around the transaction: what the sender just received, what the receiver passes on
    out_v = _sorted_values(s, t, usd)
    in_v = _sorted_values(r, t, txs["usd_out"].to_numpy())
    for W in (900, 3600, 21600):
        got = _window_sum(in_ev, in_v, s, t, W, 0)
        sent = _window_sum(out_ev, out_v, r, t, 0, W)
        f[f"s_in_usd_ratio_{W}"] = np.log1p(got) - np.log1p(usd)
        f[f"r_out_usd_ratio_{W}"] = np.log1p(sent) - np.log1p(usd)

    # pair novelty and recurrence over the week
    pair = pd.Series(list(zip(s, r)))
    pair_n = pair.map(pair.value_counts()).to_numpy()
    rev = pd.Series(list(zip(r, s))).map(pair.value_counts()).fillna(0).to_numpy()
    f["pair_n"] = pair_n
    f["pair_rev_n"] = rev
    f["pair_is_new"] = (pair_n == 1).astype(float)
    order = np.argsort(t, kind="stable")
    rank = np.empty(len(t))
    rank[order] = pair.iloc[order].groupby(pair.iloc[order]).cumcount().to_numpy()
    f["pair_rank"] = rank

    new = pair_n == 1
    nov_ev = _events(np.r_[s[new], r[new]], np.r_[t[new], t[new]])
    for W in NOVEL_WINDOWS:
        a = _window_counts(nov_ev, s, t, W, W)
        b = _window_counts(nov_ev, r, t, W, W)
        f[f"novel_s_{W}"] = a - new
        f[f"novel_r_{W}"] = b - new
        f[f"novel_both_{W}"] = a + b - 2 * new

    # how each wallet usually behaves over the week
    s_out_n = pd.Series(s).map(pd.Series(s).value_counts()).to_numpy()
    r_in_n = pd.Series(r).map(pd.Series(r).value_counts()).to_numpy()
    s_in_n = pd.Series(s).map(pd.Series(r).value_counts()).fillna(0).to_numpy()
    r_out_n = pd.Series(r).map(pd.Series(s).value_counts()).fillna(0).to_numpy()
    f["s_out_week"], f["s_in_week"], f["r_in_week"], f["r_out_week"] = s_out_n, s_in_n, r_in_n, r_out_n
    cp_s = pd.DataFrame({"a": s, "b": r}).groupby("a").b.nunique()
    cp_r = pd.DataFrame({"a": r, "b": s}).groupby("a").b.nunique()
    f["s_distinct_dst"] = pd.Series(s).map(cp_s).to_numpy()
    f["r_distinct_src"] = pd.Series(r).map(cp_r).to_numpy()
    lu = np.log1p(usd)
    s_med = pd.Series(lu).groupby(s).transform("median").to_numpy()
    s_std = pd.Series(lu).groupby(s).transform("std").fillna(1).to_numpy() + 1e-3
    r_med = pd.Series(lu).groupby(r).transform("median").to_numpy()
    r_std = pd.Series(lu).groupby(r).transform("std").fillna(1).to_numpy() + 1e-3
    f["amt_z_sender"] = (lu - s_med) / s_std
    f["amt_z_receiver"] = (lu - r_med) / r_std

    # the transaction itself
    f["log_usd"] = lu
    f["spread"] = txs["usd_out"].to_numpy() / np.maximum(usd, 1e-9) - 1
    f["n_legs"] = txs["n_legs"].to_numpy()
    f["is_conversion"] = txs["is_conversion"].to_numpy().astype(float)
    f["out_is_xrp"] = (txs["asset_out"].to_numpy() == "XRP").astype(float)
    f["in_is_xrp"] = (txs["asset_in"].to_numpy() == "XRP").astype(float)
    f["hour"] = (t % 86400) / 3600
    s_home = pd.Series(txs["asset_out"].to_numpy()).groupby(r).transform(lambda x: x.value_counts().index[0]).to_numpy()
    f["r_gets_home_asset"] = (txs["asset_out"].to_numpy() == s_home).astype(float)
    return f.astype(float)
