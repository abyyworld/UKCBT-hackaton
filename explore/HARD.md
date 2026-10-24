# Hard relay cases: what we tried (signal hunt, 24 Oct)

Result: **no module kept.** Best try `hardrelay` (hub pass-through features per payment and per pair):
`--modules local,episodes,isolation,patterns,shape,hardrelay`, seed 0: **ALL 60.01, hard 0.92**
against the base (same modules without it) **ALL 61.23, hard 0.89**. Relay family AP stayed at 1 and 2.
Dropped as the rules say. The code is not pushed.

## How the relay families are built (train labels, analysis only)

- 1,200 relay scenarios in 12 cohorts (6 per family). Each cohort is **100 copies of the same template**
  (same size, for example 100 scenarios of 26 payments), and **exactly one copy per cohort is suspicious**.
  So there are only 12 suspicious schemes in train, and every payment of a suspicious scheme is suspicious.
- A scheme is one hub with 3 to 5 inflow senders, 1 to 6 outflow receivers and 2 or 3 extra hops, active in
  episodes on 4 days.
- A sender-receiver pair almost always belongs to one scheme (12,499 pairs in one scheme, 173 in more).
- Hubs are busy: about 105 other payments in the week, including about 2 relay schemes where the wallet is the
  hub and many where it is a counterparty, plus daily background payroll pairs.

## What separates the suspicious scheme, if you know the scheme

Computed on the true scheme groups (rank of the suspicious copy within its cohort, 0 = lowest):

| Scheme signal | Suspicious copies | Normal copies | AUC |
|---|---|---|---|
| Median cover: each hub payout / (recent scheme inflow not yet paid out) | 0.16 to 0.82, always below 1 | median 2.0, often above 10 | 0.79 |
| Median gap from last inflow to payout | 2 to 5 hours | median 33 hours | 0.76 |
| Both, as a mean cohort rank | rank 0.08 to 0.36 in all 12 cohorts | | 0.80 |
| Among schemes with cover below 0.85 (467 schemes, all 12 suspicious): std of out/in per batch | 0.21 | 0.42 | 0.76 |

In words: a suspicious hub only forwards money it has just received, within a few hours, and keeps a steady share
of each batch. A normal relay hub often pays out more than its scheme brought in and waits a day or more.
Filter "cover below 0.85" keeps all 12 suspicious schemes and removes 60% of normal ones (about 2.5 times lift).

## Why it did not transfer to payments

We cannot see scheme ids in test, and the hub's other activity drowns the scheme:
- Payment level (`hardrelay`): cover and gap computed from all of the hub's non-routine payments (pairs with at
  most 4 payments) in an 8 hour window. Best within-relay AUC 0.60 (max cover over the pair), the rest 0.40 to 0.55.
  Hub-level averages: 0.43 to 0.62.
- Rebuilding schemes by linking pairs at the same hub that occur within 1 to 6 hours of each other, on 1 or 2 days:
  either impure (one cluster holds 6 schemes, purity 0.15 to 0.25) or fragmented (purity 0.73 but only 39% of a
  scheme in its main cluster). Linking across both ends of a pair percolates into one giant group.

## If someone has time later

The scheme-level signal is real (12 out of 12 cohorts) but needs scheme reconstruction first. Ideas not tried:
a learned pair-to-pair link model (train on scenario ids as the target, which is allowed as a grouping target but
risky), or matching each hub payout to the specific inflows it could be paid from (a small assignment problem per hub)
and keeping only payouts that are fully covered by inflows from 1 to 6 hours earlier.
