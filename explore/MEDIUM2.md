# Medium round 2: slow motif reconstruction and relay hub features (both dropped)

All numbers: out-of-fold AP x100, grouped 5-fold, mean of seeds 0 to 3, LightGBM with n_jobs 1.
Base = `local,episodes,isolation,patterns,shape` (stage 1).

| Version | Seeds (ALL) | Mean ALL | easy | medium |
|---|---|---|---|---|
| base | 59.50 61.47 57.72 59.86 | 59.63 | 78.83 | 8.46 |
| base + slowmotif | 59.90 61.98 58.44 60.32 | 60.16 | 79.67 | 7.29 |
| base + hubs (full, 29 features) | 59.48 61.22 57.81 59.39 | 59.47 | 79.01 | 7.96 |
| base + hubs (slim, 5 features) | 60.15 61.25 57.56 59.67 | 59.66 | 78.89 | 7.64 |

Medium per seed: base 8.44 8.91 7.11 9.38; slowmotif 7.14 8.23 6.96 6.84 (lower on every seed).
None meets the rule (medium up clearly, ALL not down), so none was pushed as a module.

## slowmotif (local slow motif around each first-time-pair payment)
From each first-time-pair payment, step to the previous and next first-time-pair payment on each of its wallets
(within T = 30 or 45 min), up to 2 or 3 steps. On that local motif: edges, wallets, closed structure
(edges minus wallets plus 1), wallet reuse, max degree, span, median and min gap, gap CV, relay wallets and
whether they receive before they send, source and sink share, recurring-partner activity of the motif wallets,
amount spread.
- Capture check on medium positives: the local motif holds 60 to 75% of the scheme with 62 to 75% purity
  (T = 45 to 60 min, 2 neighbours per side). So the reconstruction itself works.
- Univariate among first-time-pair payments (base rate 1.6%): closed structure AP 7.8%, gap CV 6.9%,
  reuse 5.4%. Best single medium signals we have seen so far.
- But in the model it adds +0.5 ALL (easy +0.8) and costs medium about 1.2. The closed structure is even more
  typical of easy motifs and fast look-alikes (closed >= 1: easy 66%, look-alike 46%, medium 35%), so the model
  uses it to push easy and look-alike groups up, and medium falls relative to them.
- Possible use: it may still be worth adding for ALL (+0.5 over 4 seeds) if the team accepts the medium cost.
  Code kept locally, can be shared on request.

## hubs (routine relay-hub activity)
Per wallet over the week: bursts of 3+ distinct counterparties within 90 min (count and days, in and out),
active days, daily CV, hour concentration and distance of this payment from the wallet's usual hour, share of
payments in this hour band, first-time-pair share, share of counterparties that recur on other days, amount
z-score against the wallet; per payment: pair active days, and the share of the wallet's counterparties within
+-1 h that recur with it on other days.
- Against the normal relay transactions ranked above the median medium positive, only two signals separate:
  counterparty recurrence around the payment (AUC 0.73; medium 0.17 vs relay 0.50) and amount z-score against
  the wallet (AUC 0.65; relay bursts are larger than the wallet's usual). Bursts, days, hour typicality and
  daily regularity are all AUC 0.45 to 0.55: medium schemes run on the same kind of busy wallets as the hubs.
- The recurrence signal is mostly already in `isolation` and the recur_* features, so the model gains nothing.
