"""Slow motifs: closure of the first-time-pair graph around a payment.

Medium schemes are the fast motifs stretched to about two hours on busy wallets. Busy wallets also make many
first-time-pair payments by coincidence, but those go to unrelated wallets: the new-pair neighbourhoods of the
two ends do not pay each other. Inside a planted motif (stack, bipartite, multi-stage, fan-out-in, cycle, chain)
the wallets around u and v are part of the same scheme, so new-pair payments link the two neighbourhoods.

For a first-time-pair payment u -> v at time t, using only first-time-pair payments within +-W:
- Nu, Nv: new-pair counterparties of u and of v (direction ignored)
- closure: payments between {u} + Nu and {v} + Nv that are not u's or v's own payments
- density: closure / (|Nu| * |Nv|), and the neighbourhood size
Recurring pairs get NaN.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

WINS = [3600, 7200, 14400, 21600]


def build(txs):
    txs = txs.reset_index(drop=True)
    n = len(txs)
    t = txs["ts"].to_numpy().astype(float)
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    pair = pd.Series(list(zip(s, r)))
    new = (pair.map(pair.value_counts()) == 1).to_numpy()
    ni = np.flatnonzero(new)
    ev = defaultdict(list)
    for i in ni:
        ev[s[i]].append((t[i], r[i], i))
        ev[r[i]].append((t[i], s[i], i))
    idx = {}
    for w, v in ev.items():
        v.sort(key=lambda e: e[0])
        idx[w] = (np.array([e[0] for e in v]), [e[1] for e in v], [e[2] for e in v])

    def nb(w, lo, hi):
        tt, oo, ii = idx[w]
        a, b = np.searchsorted(tt, lo), np.searchsorted(tt, hi, "right")
        return oo[a:b], ii[a:b]

    f = pd.DataFrame(index=txs["tx"].to_numpy())
    for W in WINS:
        clo = np.full(n, np.nan)
        dens = np.full(n, np.nan)
        size = np.full(n, np.nan)
        for i in ni:
            lo, hi = t[i] - W, t[i] + W
            ou, iu = nb(s[i], lo, hi)
            ov, iv = nb(r[i], lo, hi)
            Nu = set(ou) - {r[i]}
            Nv = set(ov) - {s[i]}
            A, B = Nu | {s[i]}, Nv | {r[i]}
            star = set(iu) | set(iv)
            edges = set()
            for x in Nu | Nv:
                ox, ix = nb(x, lo, hi)
                for o, k in zip(ox, ix):
                    if k not in star and o != x and ((x in A and o in B) or (x in B and o in A)):
                        edges.add(k)
            clo[i] = len(edges)
            dens[i] = len(edges) / max(1, len(Nu) * len(Nv))
            size[i] = len(Nu) + len(Nv)
        f[f"clo{W}"] = clo
        f[f"dens{W}"] = dens
        f[f"nb{W}"] = size
    return f.astype(float)
