# Partner notes

## Task 0: baseline
`python -m src.cv --tag baseline`: **ALL 58.59, easy 77.50, medium 5.81, hard 0.81** (matches the expected numbers).

## Task 1: relay hub features (dropped)
`src/features/relay.py` (receive and send bursts per hub, counterparty recurrence across bursts, Jaccard overlap of
sender sets, asset share, forward ratio within 12 h, amount against hub and burst median):
**ALL 58.12, easy 77.92, medium 5.28, hard 0.68**. Lower than the baseline, so it was deleted as the rules say.
Relay family AP stayed at 1, which is chance (1% of relay transactions are suspicious).

What we checked on the 200 hard relay transactions against the 19,800 normal ones (AUC within the relay families):

| Signal | AUC |
|---|---|
| Pair recurrence (count, distinct days) | 0.48 |
| Other recurring senders to the same receiver within 90 min | 0.44 |
| Other recurring recipients of the same sender within 90 min | 0.55 |
| Multi-hop chain (pays on within 3 h, two hops down or up) | 0.50 to 0.54 |
| Same 3-wallet path repeated on several days | 0.57 |
| Amount trend within pair, sender or receiver | 0.46 to 0.55 |
| Asset in or out is XRP, conversion | about 0.5 (shares nearly equal) |
| Hour, weekday, fee, sequence, flags, memo, tags | 0.49 to 0.55 |
| Wallet touches a non-relay suspicious wallet | no difference (both 92%) |

Findings:
- The 136 wallets in suspicious relays are ordinary wallets: every one also has normal transactions, and only about
  8% of each wallet's transactions are suspicious. Wallet-level flags cannot find them.
- Normal relays are built from the same template (recurring senders in bursts, forwards a few hours later), so the
  shape alone does not separate them. Prototype: `explore/relay_chain.py`.
- Best lead: the repeated 3-wallet path (`path_days`, AUC 0.57). Weak, but the only one above 0.55.

Suggestion: spend the remaining time on medium (5.8) rather than hard relays. Medium cases are 18% of positives and
look far more learnable than the hard relays.

## Task 2: train vs test shift
See `explore/ADV_VAL.md`. No feature shifts on its own (largest single-feature AUC 0.548). The 0.99 combined AUC is a
wallet fingerprint from the weekly count features, not a behaviour change.
