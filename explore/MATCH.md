# Value matching among first-time pairs (helper 3, medium cases)

Idea: laundering passes value on almost whole, so for a first-time-pair payment s -> r, look for an onward
first-time-pair payment out of r (or an inflow into s just before) whose amount matches. Busy-wallet noise should
not match amounts. Also collectors (one outflow matches the sum of recent inflows), splitters (outflows sum to one
inflow), and how many hops the best matches can be followed.

Module was `src/features/match.py` (deleted, see below). 3-seed comparison: `python -m explore.match_cv <modules> ...`.

## What medium value flow really looks like (train, labels used only for analysis)

- Only 105 of the 225 medium suspicious payments have an onward payment out of the receiver inside the same
  scenario. Fan-out leaves, fan-in sources and the last hop of a chain never pass on.
- Value is not passed on whole. Per hop, received to passed-on ratio (median, IQR):
  chain 0.91 (0.89 to 1.01), cycle 0.95 (0.86 to 0.97), fan_out_in 0.87 (0.79 to 1.04),
  random 0.74 (0.62 to 0.89), stack 0.72 (0.48 to 1.61), multi_stage 1.93 (0.81 to 3.95, collectors merge many).
- Delays: chain about 11 min, cycle about 27 min, fan_out_in about 53 min, stack and multi-stage 60 to 90 min,
  95th percentile about 4 h.
- So a tight band (0.85 to 1.02 within 90 min) matches only 10% of medium payments. v1 uses a band of 0.7 to 1.15
  around a typical kept share of 0.92, with windows of 1 h and 3 h.

## Features tried (v1)

For each first-time-pair payment, using only other first-time-pair payments, at windows of 1 h and 3 h:
number of onward candidates out of the receiver, number inside the band, best ratio, its log error against 0.92,
its delay; the same backwards into the sender; how many hops the best in-band matches can be followed forwards and
backwards (chain length). Collector: an outflow of the receiver within 2 h matching the sum of a run of its inflows
(this one included). Splitter: a run of the sender's outflows (this one included) matching one inflow before.

Single-feature view, medium suspicious against normal:
- Best-match log error (1 h) has AUC 0.70 (inverted), but it exists for only 32% of medium and 6% of normal payments.
- Chain length 4 or more at 1 h: 17 medium against 76 normal (18% precision against a 0.6% base). These are mostly
  chains and cycles, which `patterns` (reach, cycle) and `shape` (kept share) already see.
- Collector and splitter sum-match flags are at chance (AUC 0.51 to 0.52), and their errors exist for only 20 to 25%
  of medium payments (AUC 0.51 to 0.61 there): stack and multi-stage amounts do not add up.

## Results (out-of-fold AP x100, grouped 5-fold, seeds 0, 1, 2)

Base = `local,episodes,isolation,patterns,shape` (v13 module set, stage 1).

| Version | Seeds ALL | Mean ALL | easy | medium | hard |
|---|---|---|---|---|---|
| base | 59.50 61.47 57.72 | **59.56** | 78.95 | **8.15** | 0.87 |
| base + match v1 (all 30 features) | 59.53 61.43 57.42 | 59.46 | 78.94 | 7.64 | 0.86 |
| base + match v2 (chain length and match errors only) | 59.51 60.83 56.69 | 59.01 | 78.01 | 7.48 | 0.88 |
| base + match v3 (chain length only, 2 features) | 60.26 61.45 57.23 | 59.65 | 79.01 | 8.00 | 0.89 |

No version reaches the bar (ALL +0.5, or medium +3 without ALL falling). **Dropped, `match.py` deleted.**

## Why it does not work

- The premise does not hold for medium cases: hops keep about 0.9 with a wide spread, and in stack, multi-stage and
  random motifs the amounts are drawn fresh at each step. A band wide enough to catch them (0.7 to 1.15) is wide
  enough for busy-wallet noise: about 7% of normal first-time-pair payments get an in-band onward match within 3 h.
- Half of medium payments (fan leaves, sources, last hops) have no onward payment to match at all.
- The motifs where matching does work (chain, cycle) are already found by `patterns` and `shape`.

## Practical note for whoever runs CV next

`python -m explore.match_cv` takes `SEEDS=0,1,2`. Running one process per seed with `LGB_PARAMS='{"n_jobs": 1}'`
in parallel takes about 2 to 3 minutes for all three seeds on 4 cores. Two processes with `n_jobs=4` each took over
20 minutes per seed because of thread oversubscription.
