# Motif shape features (`src/features/shape.py`)

Idea: link first-time sender-receiver pairs that share a wallet within T seconds (T = 60, 300, 900), then describe
how money flows through each linked group: value kept by pass-through wallets, delay before passing on, hop to hop
amount change, how regular the gaps are, share of sources, sinks and pass-through wallets, end-to-end conservation,
number of assets, group amount against the wallets' usual amount, and how busy the wallets are outside the group
(within 6 h). Per transaction: did the receiver pass the money on inside the group, how fast, how much it kept,
and how busy sender and receiver are outside the group. Group size, span and median gap are included too.

## Results (out-of-fold AP x100, grouped 5-fold, mean of seeds 0, 1, 2)

`python -m explore.shape_cv <modules> ...`

| Version | Seeds | Mean ALL | easy | medium |
|---|---|---|---|---|
| local | 59.53 60.41 56.86 | 58.93 | 77.24 | 7.37 |
| local + shape v1 (T = 120, 600, no tempo) | 60.00 60.78 56.81 | 59.19 | 77.96 | 6.34 |
| local + shape v2 (T = 60, 300, 900, no tempo) | 60.22 60.64 57.50 | 59.45 | 78.36 | 6.22 |
| **local + shape v3 (v2 + size, span, gap median)** | 59.84 61.79 58.50 | **60.04** | 79.42 | 6.27 |
| local + episodes | 58.29 58.95 55.87 | 57.70 | 76.56 | 7.45 |
| **local + episodes + shape v3** | 59.64 60.99 58.12 | **59.58** | 79.48 | 7.53 |

Kept: +1.11 over local, +1.88 over local + episodes. The gain is on the easy cases. Medium drops a little against
local alone, but not against local + episodes.

## Findings

- Inside fast scenarios only (span 20 min or less), easy suspicious vs look-alikes, the best single signals are
  value conservation: suspicious pass-through wallets forward more of what they receive, more evenly
  (`n60_kept_std`, `kept_med`, `e2e_kept`, AUC 0.62 to 0.69). Max in-degree and number of pass-through wallets
  help a little (0.60 to 0.64).
- Fast look-alikes really do look like the planted motifs: same structure, same amounts, same assets, single-leg
  payments, same fee, `last_ledger_gap` always 20, same amount precision. Sequence gaps, flags, tags, memo, wallet
  degree, wallet novelty share, active days: all AUC 0.45 to 0.51. Wallets taking part in many other fast groups:
  0.49 to 0.63, weak.
- Activity outside the group within 6 h is only weakly lower for suspicious wallets (AUC about 0.57).
- So the gain comes mostly from describing the first-time-pair group well (tempo plus flow), not from one strong
  separator. The roughly 30 fast look-alike scenarios stay hard.

Scripts: `explore/shape_cv.py` (3-seed comparison), `explore/shape_auc.py` (per-feature AUC, easy vs look-alikes),
`explore/shape_fast.py` (same inside fast scenarios only). Labels used only for analysis.

## Follow-up: on top of Aby's newest features, and shape v4

Base = `local,episodes,isolation,patterns` (after commits 42bbf94 and 7620c9d). Mean of seeds 0, 1, 2.

| Version | Seeds | Mean ALL | easy | medium |
|---|---|---|---|---|
| local + episodes + isolation + patterns | 57.52 59.52 56.43 | 57.82 | 76.09 | 8.23 |
| **same + shape v3** | 59.50 61.47 57.72 | **59.56** | 78.95 | 8.15 |
| same + shape v4 | 59.23 61.05 58.17 | 59.48 | 78.74 | 8.08 |
| local + shape v4 (vs v3 60.04) | 60.76 62.63 58.54 | 60.64 | 80.19 | 6.69 |

- Shape v3 still adds **+1.74** on Aby's newest set. It is not redundant with `patterns`: no shape feature has rank
  correlation above 0.68 with any patterns feature (the closest are sources and sinks share, max in-degree and
  outside activity at T = 60).
- Shape v4 added per-payment conservation at T = 60 and 300: what the receiver passes on after this payment (kept
  fraction, delay, split evenness), what the sender received just before (kept fraction, delay), conservation
  over the 2-3 hop path, and the sender's split evenness (outgoing sum against incoming sum, CV of amounts).
  It beats v3 by +0.60 on local alone, but gives nothing on Aby's set (59.48 vs 59.56): `patterns` already covers
  the path and scatter-gather view. **Dropped; v3 stays.**

Suggestion for Aby: add `shape` (v3, as committed) to the module list of stage 1 and stage 2.
