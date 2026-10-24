# Shift audit at the top of the ranking (train vs test)

Scripts: `explore/shift_audit_build.py` (caches features), `explore/shift_audit.py` (the audit, full table in
`data/shift_audit.csv`), `explore/shift_drop_cv.py` (out-of-fold AP when dropping a group).
Modules: local, episodes, isolation, patterns, shape, slow (459 features).

Motif-like rows: episode `all300` has 4 to 60 payments and at least half are first-time pairs.
Train 2,773 rows (29% suspicious), test 2,165 rows. For each feature: KS train vs test on these rows, KS train
suspicious vs train normal, and `shift` = (test median - train median) / (train suspicious median - train normal
median). Negative shift means test moves toward the train normal class, so test laundering would look more normal.
Risk = KS(train, test) x KS(classes) when the shift is toward normal.

## Problem 1: amount evenness (same as the partner's finding)

Every top-risk feature that is not a giant-episode artefact measures how even or matching the amounts are. In train,
suspicious groups are even; on test motif rows these features sit at or beyond the train normal level.

| Feature | Train suspicious | Train normal | Test | KS train vs test | KS classes |
|---|---|---|---|---|---|
| episodes__novel3600_drift_abs | 0.90 | 1.30 | 1.40 | 0.45 | 0.45 |
| episodes__novel3600_usd_std | 1.00 | 1.17 | 1.33 | 0.49 | 0.40 |
| episodes__novel3600_usd_range | 3.18 | 3.35 | 3.93 | 0.24 | 0.36 |
| episodes__all600_usd_std | 0.79 | 1.10 | 1.11 | 0.22 | 0.38 |
| episodes__all600_drift_abs | 0.72 | 1.21 | 1.14 | 0.20 | 0.37 |
| episodes__all300_usd_std | 0.67 | 0.91 | 0.92 | 0.14 | 0.30 |
| episodes__all30_usd_std | 0.28 | 0.44 | 0.52 | 0.18 | 0.25 |
| shape__n60_kept_std | 0.10 | 0.28 | 0.11 (train 0.11) | 0.22 | 0.42 |
| shape__n300_kept_std | 0.12 | 0.37 | 0.16 (train 0.12) | 0.17 | 0.41 |
| shape__n300_hop_ratio_abs | 0.53 | 0.63 | 0.73 | 0.16 | 0.22 |
| shape__n900_hop_ratio_abs | 0.57 | 0.83 | 0.83 | 0.13 | 0.30 |

The same pattern holds for the other `usd_std`, `drift_abs`, `usd_range` scales (all30 to all600, novel1800,
novel3600). `kept_std` moves less but in the same direction.

## Problem 2: giant first-time-pair episodes are a split fingerprint

At the coarse first-time-pair scales the episodes percolate into one giant component:

| Scale | Train giant size, share of rows | Test giant size, share of rows |
|---|---|---|
| novel3600 | 3,326 (8%) | 2,096 (5%) |
| novel7200 | 4,356 (11%) | 4,361 (11%) |
| novel21600 | 14,005 (35%) | 14,048 (35%) |

Every row inside the giant component gets the same episode statistics, a constant per split. The constants differ
between train and test (for example novel21600 usd_std 1.27 vs 1.40, xrp_out 0.28 vs 0.37, conv 0.42 vs 0.50), so
these features have KS 0.7 to 1.0 between splits while carrying no class signal inside the giant group (KS classes
about 0). A tree split that falls between the two constants sends a third of test to the other branch.
Also shifted outside the giant group: `xrp_out` and `conv` at novel3600 and novel7200 (test has more XRP out,
0.35 to 0.37 vs 0.22 to 0.23), and `novel7200_gap_med` (2.8 train vs 3.6 test).

## Cost of dropping (out-of-fold AP, seed 0, same six modules)

Running (sequential, seed 0). Numbers will be added here.

## Regex of feature names to drop

```
^episodes__novel(7200|21600)_|(usd_std|usd_range|drift_abs|kept_std|kept_absmax|hop_ratio_abs)$
```

(76 features: 46 giant-scale episode features and 36 amount-evenness features, 6 overlap.)

## Recommendation

1. Drop the giant-scale episode features first (`^episodes__novel(7200|21600)_`). They carry no class signal
   inside the giant group and are a pure split fingerprint, so the leaderboard risk is high and the OOF cost should
   be small. A softer fix inside `episodes.py` (Aby's file): set the episode stats to NaN when the episode has more
   than about 200 payments, at every scale, which also covers the novel3600 giant group (8% of train rows).
2. Drop or neutralise the amount-evenness group
   (`(usd_std|usd_range|drift_abs|kept_std|kept_absmax|hop_ratio_abs)$`). Expect OOF to fall, because train
   rewards even amounts, but the leaderboard should gain or hold, as the partner saw with v17 (+3.7 OOF, +0.5 LB).
   Judge this one by an upload, not by OOF.
3. Keep timing, structure, counterparty and flow-balance features: their test medians sit inside the train range
   (for example gap, span, size, wallet counts, density at the tight scales).
