"""Graph pattern counts in the style of IBM's Graph Feature Preprocessor (GFP), over all payments.

From the AMLworld paper setup: scatter-gather within 6 hours, temporal cycles within 1 day.
For every payment u -> v at time t:
- scatter-gather as first leg: over two-hop paths u -> v -> w with the second leg within (t, t + 6h], how many
  distinct middles connect u and w (k); count of such w with k >= 2, and the max k.
- scatter-gather as second leg: over x -> u -> v with the first leg within [t - 6h, t), same counts for (x, v).
- gather-scatter through v: distinct senders into v in [t - 6h, t] times distinct receivers out of v in (t, t + 6h].
- temporal cycles within 1 day: v -> u after t (length 2), and v -> x -> u with increasing times (length 3).
- first-contact order ("port numbering"): rank of v among u's distinct receivers by first payment, and rank of u
  among v's distinct senders, plus how many new counterparties each made before this payment.
"""
import bisect
from collections import defaultdict

import numpy as np
import pandas as pd

SG_WINDOW = 21600
CYCLE_WINDOW = 86400


def _lists(keys, times, others):
    d = defaultdict(list)
    for k, ti, o in zip(keys, times, others):
        d[k].append((ti, o))
    for k in d:
        d[k].sort()
    return d


def _slice(lst, lo, hi, lo_open=False, hi_open=False):
    i = bisect.bisect_right(lst, (lo, "￿")) if lo_open else bisect.bisect_left(lst, (lo, ""))
    j = bisect.bisect_left(lst, (hi, "")) if hi_open else bisect.bisect_right(lst, (hi, "￿"))
    return lst[i:j]


def build(txs):
    txs = txs.reset_index(drop=True)
    n = len(txs)
    t = txs["ts"].to_numpy().astype(float)
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    out_adj = _lists(s, t, r)   # wallet -> [(time, receiver)]
    in_adj = _lists(r, t, s)    # wallet -> [(time, sender)]
    cols = {k: np.zeros(n) for k in [
        "sg1_k2", "sg1_kmax", "sg2_k2", "sg2_kmax", "gs_v", "gs_u", "cyc2", "cyc3"]}
    W = SG_WINDOW
    for i in range(n):
        u, v, ti = s[i], r[i], t[i]
        # scatter-gather, this payment as first leg: u -> v -> w
        outs_u = {}
        for tm, m in _slice(out_adj.get(u, []), ti - W, ti + W):
            outs_u.setdefault(m, tm)
        best, cnt = 0, 0
        for tw, w in _slice(out_adj.get(v, []), ti, ti + W, lo_open=True):
            k = 0
            for tm2, m2 in _slice(in_adj.get(w, []), ti - W, tw + W):
                if m2 in outs_u:
                    k += 1
            best = max(best, k)
            cnt += k >= 2
        cols["sg1_k2"][i], cols["sg1_kmax"][i] = cnt, best
        # scatter-gather, this payment as second leg: x -> u -> v
        ins_v = {a for _, a in _slice(in_adj.get(v, []), ti - W, ti + W)}
        best, cnt = 0, 0
        for tx_, x in _slice(in_adj.get(u, []), ti - W, ti, hi_open=True):
            k = sum(1 for _, m in _slice(out_adj.get(x, []), tx_ - W, ti + W) if m in ins_v)
            best = max(best, k)
            cnt += k >= 2
        cols["sg2_k2"][i], cols["sg2_kmax"][i] = cnt, best
        # gather-scatter through v and through u
        cols["gs_v"][i] = (len({a for _, a in _slice(in_adj.get(v, []), ti - W, ti)})
                           * len({b for _, b in _slice(out_adj.get(v, []), ti, ti + W, lo_open=True)}))
        cols["gs_u"][i] = (len({a for _, a in _slice(in_adj.get(u, []), ti - W, ti, hi_open=True)})
                           * len({b for _, b in _slice(out_adj.get(u, []), ti, ti + W)}))
        # temporal cycles within a day
        back = _slice(out_adj.get(v, []), ti, ti + CYCLE_WINDOW, lo_open=True)
        cols["cyc2"][i] = sum(1 for _, b in back if b == u)
        c3 = 0
        for tx1, x in back:
            if x == u:
                continue
            c3 += sum(1 for _, b in _slice(out_adj.get(x, []), tx1, ti + CYCLE_WINDOW, lo_open=True) if b == u)
        cols["cyc3"][i] = c3
    f = pd.DataFrame(cols, index=txs["tx"].to_numpy())
    # first-contact order (port numbering)
    order = np.argsort(t, kind="stable")
    d = pd.DataFrame({"s": s[order], "r": r[order]})
    first_pair = ~d.duplicated(["s", "r"])
    d["new_for_s"] = first_pair.groupby(d["s"]).cumsum()
    d["new_for_r"] = first_pair.groupby(d["r"]).cumsum()
    pair_rank_s = d[first_pair].assign(k=lambda x: x.groupby("s").cumcount()).set_index(["s", "r"])["k"]
    pair_rank_r = d[first_pair].assign(k=lambda x: x.groupby("r").cumcount()).set_index(["s", "r"])["k"]
    idx = pd.MultiIndex.from_arrays([s, r])
    f["port_rank_in_sender"] = pair_rank_s.reindex(idx).to_numpy()
    f["port_rank_in_receiver"] = pair_rank_r.reindex(idx).to_numpy()
    back_order = np.empty(n, int)
    back_order[order] = np.arange(n)
    f["new_partners_s_so_far"] = d["new_for_s"].to_numpy()[back_order]
    f["new_partners_r_so_far"] = d["new_for_r"].to_numpy()[back_order]
    return f.astype(float)
