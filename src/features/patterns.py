"""Explicit laundering pattern counts among first-time pairs, over windows of hours.

Slow (medium) schemes are built from brand-new wallet pairs. Normal life rarely forms exact shapes out of new
pairs, so for each first-time-pair payment u -> v at time t we count, within +-W seconds and using only other
first-time-pair payments:
- cycle: a time-respecting path from v back to u (length and whether it exists)
- reach: how far money can travel forward from v and backward into u (nodes and depth)
- scatter-gather: parallel two-hop paths that share this payment's ends
- bipartite block: density of new pairs between the senders into v and the receivers out of u
Payments between pairs that recur in the week get NaN (the model treats them as missing).
"""
from collections import defaultdict
from heapq import heappop, heappush

import numpy as np
import pandas as pd

WINDOWS = [1800, 3600, 7200, 21600, 43200]
MAX_DEPTH = 8


def _adjacency(u, v, t):
    out_adj, in_adj = defaultdict(list), defaultdict(list)
    for a, b, ti in zip(u, v, t):
        out_adj[a].append((ti, b))
        in_adj[b].append((ti, a))
    for d in (out_adj, in_adj):
        for k in d:
            d[k].sort()
    return out_adj, in_adj


def _window(lst, lo, hi):
    """Entries of a time-sorted list with lo <= time <= hi."""
    import bisect
    i = bisect.bisect_left(lst, (lo, ""))
    j = bisect.bisect_right(lst, (hi, "￿"))
    return lst[i:j]


def _earliest_arrival(adj, start, t0, t_end, forward=True):
    """Time-respecting search from start beginning at t0. Returns {node: (time, depth)}."""
    best = {start: (t0, 0)}
    heap = [(t0, 0, start)]
    while heap:
        tc, depth, node = heappop(heap)
        if depth >= MAX_DEPTH or best.get(node, (np.inf,))[0] < tc:
            continue
        lo, hi = (tc, t_end) if forward else (t_end, tc)
        for tn, nxt in _window(adj.get(node, []), lo, hi):
            if (forward and tn < tc) or (not forward and tn > tc):
                continue
            key = tn if forward else -tn
            prev = best.get(nxt)
            if prev is None or (key < (prev[0] if forward else -prev[0])):
                best[nxt] = (tn, depth + 1)
                heappush(heap, (key if forward else key, depth + 1, nxt))
    return best


def build(txs):
    txs = txs.reset_index(drop=True)
    t = txs["ts"].to_numpy().astype(float)
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    pair = pd.Series(list(zip(s, r)))
    new = (pair.map(pair.value_counts()) == 1).to_numpy()
    out_adj, in_adj = _adjacency(s[new], r[new], t[new])
    f = pd.DataFrame(index=txs["tx"].to_numpy())
    rows = np.flatnonzero(new)
    for W in WINDOWS:
        cols = defaultdict(lambda: np.full(len(txs), np.nan))
        for i in rows:
            u, v, ti = s[i], r[i], t[i]
            # forward reach from v and whether it returns to u (cycle)
            fwd = _earliest_arrival(out_adj, v, ti, ti + W, forward=True)
            cols["reach_fwd"][i] = len(fwd) - 1
            cols["depth_fwd"][i] = max(d for _, d in fwd.values())
            cols["cycle_len"][i] = fwd[u][1] + 1 if u in fwd else 0
            # backward reach into u
            bwd = _earliest_arrival_backward(in_adj, u, ti, ti - W)
            cols["reach_bwd"][i] = len(bwd) - 1
            cols["depth_bwd"][i] = max(d for _, d in bwd.values())
            # scatter-gather: other middles m with u -> m -> w where v -> w (this payment is the first hop)
            outs_u = {b for _, b in _window(out_adj.get(u, []), ti - W, ti + W)}
            sg1 = 0
            for _, w in _window(out_adj.get(v, []), ti - W, ti + W):
                ins_w = {a for _, a in _window(in_adj.get(w, []), ti - W, ti + W)}
                sg1 += len((outs_u & ins_w) - {v})
            cols["sg_first_hop"][i] = sg1
            # this payment as the second hop: sources x -> u, other middles m with x -> m -> v
            ins_v = {a for _, a in _window(in_adj.get(v, []), ti - W, ti + W)}
            sg2 = 0
            for _, x in _window(in_adj.get(u, []), ti - W, ti + W):
                outs_x = {b for _, b in _window(out_adj.get(x, []), ti - W, ti + W)}
                sg2 += len((outs_x & ins_v) - {u})
            cols["sg_second_hop"][i] = sg2
            # bipartite block: senders into v (S) and receivers out of u (R), density of new pairs S -> R
            S, R = ins_v, outs_u
            edges = sum(len({b for _, b in _window(out_adj.get(a, []), ti - W, ti + W)} & R) for a in S)
            cols["block_senders"][i] = len(S)
            cols["block_receivers"][i] = len(R)
            cols["block_density"][i] = edges / max(len(S) * len(R), 1)
        for k, arr in cols.items():
            f[f"{k}_{W}"] = arr
    return f.astype(float)


def _earliest_arrival_backward(in_adj, start, t0, t_start):
    """Time-respecting search backwards in time: who could have funded start before t0."""
    best = {start: (t0, 0)}
    heap = [(-t0, 0, start)]
    while heap:
        neg, depth, node = heappop(heap)
        tc = -neg
        if depth >= MAX_DEPTH or best.get(node, (-np.inf,))[0] > tc:
            continue
        for tp, prev_node in _window(in_adj.get(node, []), t_start, tc):
            if tp > tc:
                continue
            old = best.get(prev_node)
            if old is None or tp > old[0]:
                best[prev_node] = (tp, depth + 1)
                heappush(heap, (-tp, depth + 1, prev_node))
    return best
