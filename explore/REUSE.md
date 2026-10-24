# Medium wallet reuse: dropped (no reliable gain)

Idea: planted schemes reuse a small set of wallets, so the new counterparties of a payment's sender and receiver
should also form new pairs with each other. Labels, `scenario_id` and `difficulty` were used for analysis only.

## What the data says
- The whole train week has only **700 wallets**. Scenario wallets are a pool of about 675 busy wallets; each one sits
  in about 24 scenarios, most of them normal look-alikes. Every easy and medium wallet is in more than one scenario.
- Medium, easy and normal scenario wallets look the same: about 109 payments a week, about 24 scenarios each.
  So wallet membership and week-wide reuse cannot separate them.
- Week-wide first-time-pair graph (common neighbours, Jaccard, squares through the payment, degrees): AUC 0.42 to
  0.61 for medium against normal. The graph is dense, so co-membership is noise.
- Time-local version works better. For a first-time-pair payment s -> r, take N(s) and N(r) = wallets s and r form
  first-time pairs with within +-W. Count first-time-pair links between N(s) and N(r) within +-W:

| W | links AUC (medium vs normal) | density AUC |
|---|---|---|
| 1 h | 0.651 | 0.651 |
| 6 h | 0.762 | 0.765 |
| 12 h | 0.724 | 0.750 |
| 24 h | 0.689 | 0.736 |

  Jaccard of N(s) and N(r) is 0.50 (no signal). Correlation with any `patterns` column is at most 0.43, so this is
  new information, but AUC 0.76 at a 0.6% base rate does not reach the top of the ranking.

## CV (grouped 5-fold, base = local, episodes, isolation, patterns, shape)

| Version | seeds | ALL | medium |
|---|---|---|---|
| base | 0, 1, 2 | 59.56 | 8.15 |
| base + reuse, all 16 columns | 0, 1, 2 | 59.74 (+0.17) | 7.37 (-0.78) |
| base + reuse lean (dens 6 h, dens 12 h, links 6 h) | 0, 1, 2 | 60.04 (+0.47) | 8.56 (+0.41) |
| base | 3, 4 | 59.63 | 8.71 |
| base + reuse lean | 3, 4 | 59.75 (+0.12) | 8.77 (+0.06) |

Lean over 5 seeds: ALL +0.33, medium +0.27. Per seed ALL: +0.97, -0.03, +0.48, +0.13, +0.11. Below the bar
(ALL +0.5 or medium +3), so `src/features/reuse.py` was deleted. The lean subset was chosen after the first run,
which is why it was re-checked on fresh seeds.

## Code (to revive it, save as `src/features/reuse.py` and add `reuse` to the modules)

```python
"""Wallet reuse among first-time pairs: do the new counterparties of the sender and of the receiver also form new
pairs with each other around the payment?

For each first-time-pair payment s -> r at time t, take N(s) and N(r), the wallets s and r form first-time pairs
with (either direction) within +-W, excluding each other. Count the first-time-pair links between N(s) and N(r)
within +-W, and the density of those links. A planted scheme reuses the same small set of wallets, so its new
counterparties link to each other; a busy normal wallet's new counterparties are mostly unrelated.
Payments between pairs that recur in the week get NaN.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd

WINDOWS = [3600, 21600, 43200, 86400]


def build(txs):
    txs = txs.reset_index(drop=True)
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    ts = txs["ts"].to_numpy().astype(float)
    pair = pd.Series(list(zip(s, r)))
    new = (pair.map(pair.value_counts()) == 1).to_numpy()
    ev = defaultdict(list)
    for i in np.flatnonzero(new):
        ev[s[i]].append((ts[i], r[i]))
        ev[r[i]].append((ts[i], s[i]))
    for w in ev:
        ev[w].sort()
    times = {w: [e[0] for e in v] for w, v in ev.items()}

    def neigh(w, lo, hi):
        return {e[1] for e in ev[w][bisect_left(times[w], lo):bisect_right(times[w], hi)]}

    idx = np.flatnonzero(new)
    f = pd.DataFrame(index=txs["tx"].to_numpy())
    for W in WINDOWS:
        out = np.full((len(txs), 4), np.nan)
        for i in idx:
            lo, hi = ts[i] - W, ts[i] + W
            ns, nr = neigh(s[i], lo, hi) - {r[i]}, neigh(r[i], lo, hi) - {s[i]}
            links = sum(len(neigh(u, lo, hi) & nr) for u in ns) if ns and nr else 0
            out[i] = [len(ns), len(nr), links, links / max(len(ns) * len(nr), 1)]
        for k, name in enumerate(["ns", "nr", "links", "dens"]):
            f[f"{name}_{W}"] = out[:, k]
    return f
```
