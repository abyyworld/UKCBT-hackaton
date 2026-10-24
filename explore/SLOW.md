# Medium slow motifs (`src/features/slow.py`)

Base for every row: `local,episodes,isolation,patterns,shape`. Out-of-fold AP x100, grouped 5-fold, mean of seeds
0, 1, 2 (`python -m explore.shape_cv <modules>`). Keep rule: ALL up by 0.5 or more, or medium up by 3 or more
with ALL not falling.

| Version | Seeds | Mean ALL | easy | medium |
|---|---|---|---|---|
| base | 59.50 61.47 57.72 | 59.56 | 78.95 | 8.15 |
| + v1 matched hops and flow neighbours | 59.37 61.31 58.36 | 59.68 | 79.13 | 7.54 |
| + v2 closure, windows 0.5, 1, 2, 4 h | 60.13 62.02 58.01 | 60.05 | 78.98 | 7.48 |
| **+ v2 closure, windows 1, 2, 4, 6 h (kept)** | 60.30 61.74 58.28 | **60.11** | 79.01 | 7.96 |
| base, medium rows weighted x3 (`W_MEDIUM=3`) | 59.45 60.75 57.40 | 59.20 | 78.38 | 7.36 |
| + v2 closure 1-6 h, `W_MEDIUM=3` | 59.59 61.97 57.43 | 59.67 | 78.14 | 7.69 |

Kept v2: +0.55 ALL. Medium AP inside the model stays at about 8.

## What v2 does
For a first-time-pair payment u -> v, take the new-pair counterparties of u (Nu) and of v (Nv) within +-W
(direction ignored). Closure is the number of new-pair payments within +-W between {u} + Nu and {v} + Nv, other than
u's and v's own. Also density (closure / |Nu| |Nv|) and neighbourhood size. Busy wallets make new pairs by
coincidence, but those go to unrelated wallets whose neighbourhoods do not pay each other. Inside a planted motif
they do.

## Findings (medium suspicious against all normal, single features)

| Signal | AUC | AP | Fires on medium | Fires on normal |
|---|---|---|---|---|
| closure, +-1 h | 0.86 | 8.9 | 31% | 0.9% |
| closure, +-2 h | 0.87 | 10.3 | 39% | 1.8% |
| closure, +-4 h | 0.76 | 12.5 | | |
| closure, +-6 h | 0.76 | 13.0 | | |
| best `patterns` feature (scatter-gather, 6 h) | 0.60 | 9.2 | | |
| value-matched forward hop (USD within 25%, 1 h) | 0.57 | 2.4 | 16% | 0.6 to 1.7% |
| flow neighbours (matched-hop payments on the same wallets, 1 h) | 0.65 | 5.3 | 44% | 6% |
| layer ordering (all new-pair ins before outs on the middle wallet, 2 h) | 0.53 | 1.7 | | |

- Closure is the best single medium signal found so far (AP 13 alone, against a 0.6% base rate). Its rank
  correlation with every `patterns` feature is 0.61 or less. But inside the full model medium AP stays at about 8:
  the trees use it mostly for easy cases.
- Value matching is too sparse: only 37 value-matched forward hops exist among all 225 medium payments. Fan-out,
  fan-in and bipartite medium motifs have none. Matched-hop delays in medium (median 29 min) look like background.
- Layer ordering hardly separates: busy wallets' coincidental new pairs also look ordered.
- Exclusive use (angle 3) does not exist here. Planted wallets are shared with other scenarios as much as normal
  look-alike wallets are (median 10 suspicious and about 140 normal other scenarios per scenario, for easy, medium
  and look-alikes alike).
- Up-weighting medium rows (`W_MEDIUM=3`) lowers both ALL and medium.

Scripts: `explore/slow_auc.py` (single-feature AUC and AP, medium against normal), `explore/shape_cv.py`.
Labels used only for analysis.

Suggestion for Aby: add `slow` to the module list. Medium may do better with a rank blend that gives the closure
score its own weight (for example within the top 1%) than with trees alone.
