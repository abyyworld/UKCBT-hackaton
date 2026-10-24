# Decode & Detect: team Batman and Robin

Fewer than 10 slides. Plain words. Placeholders are marked TODO.

---

## 1. The problem

- A week of synthetic payments on a fictional XRP-style ledger. Some of it was planted to look like money laundering.
- Task: give every test transaction a suspicion score from 0 to 1 so the suspicious ones come first.
- Score: average precision. It rewards putting real cases at the top of an analyst's review queue.
- We are scoring **transactions flagged for review**, not accusing any address.

---

## 2. The data

- Train: 40,131 labelled transactions. Test: 40,101 unlabelled. Seven days each. Wallets do not overlap between splits.
- Each transaction has 1 to 3 transfer legs. Conversions go through liquidity providers.
- Cleaning: XRP drops to XRP, assets keyed by issuer and symbol, every payment valued in USD using the hour's
  reference price, one row per transaction.
- Suspicious cases: 64% easy, 18% medium, 18% hard. The hard ones hide inside about 20,000 normal relay hub payments.

---

## 3. What makes laundering visible here

- **First-time pairs:** almost every step of a planted motif (98%) is the first payment between those two wallets.
  But so are 55% of normal payments, so novelty alone is not enough.
- **Tempo is the main signal.** Typical time between steps inside a scenario (train, median):

| | Easy suspicious | Normal look-alike | Medium suspicious |
|---|---|---|---|
| Gap between steps | about 1 min | about 10 min | about 18 min |
| Whole motif | about 5 min | about 60 min | about 110 min |

- Normal look-alikes (marketplaces, payroll, treasury) build the same shapes but at a business pace. Easy laundering
  is much faster; medium laundering is deliberately slower. So the useful signal is "unusual tempo for this shape",
  not simply "fast".
- Everything else (amount, conversions, asset, fees, hour, wallet age) is almost identical between suspicious and
  look-alike scenarios.

---

## 4. Features

- Local: amounts in USD, conversions, number of legs, sender and receiver activity in several time windows.
- Flow: pass-through time, share of value kept, fan-in and fan-out counts.
- Relay hubs: burst size and duration, sender recurrence across bursts, in/out balance. TODO: keep only if it helped.
- TODO: top 10 features by importance.

---

## 5. Model and validation

- LightGBM gradient-boosted trees.
- Grouped 5-fold cross-validation: every planted scenario stays inside one fold, so the score is not leaked.
- Robustness: a train-vs-test classifier to spot features that shift between splits. TODO: AUC and what we dropped.
- No labels, scenario IDs or split names used as features.

---

## 6. Results

| Version | Out-of-fold AP | Easy | Medium | Hard | Leaderboard |
|---|---|---|---|---|---|
| Local features only (baseline) | 58.6 | 77.5 | 5.8 | 0.8 | |
| Local + episodes, stage 2 group scoring | 61.6 | | | | 67.3 |
| + scheme isolation | 61.9 | | | | 67.6 (blend) |
| v13: + first-time-pair patterns + motif shape, stage 2 | 63.8 | 79.2 | 10.0 | 2.1 | TODO |
| Final | TODO | | | | TODO |

- The leaderboard runs about 5.7 points above our out-of-fold AP; differences under 1 point are noise.
- Ideas tested and dropped because they did not raise out-of-fold AP: relay hub features, slow-motif window counts,
  closed squares, a medium-only model, pseudo-labelling, wider pattern windows (see `explore/NOTES.md`).

---|---|---|---|---|---|
| Baseline | 58.6 | 77.5 | 5.8 | 0.8 | TODO |
| TODO | | | | | |
| Final | TODO | | | | TODO |

---

## 7. Flagged for review: examples

Details and full hashes in `report/EXAMPLES.md`.

- **Rank 1, `5736BA1F…`:** one wallet splits about $127 across six new wallets in 1.7 minutes; all six pay one
  collector within the next 2 minutes. Twelve payments, all new pairs.
- **Rank 4, `8AA1DCA1…`:** the same split and merge, but each branch converts USD into EUR or GBP and back to USD.
  98% to 100% of the value survives each hop, all within 10 minutes.
- **Rank 5, `41EF2ECA…`:** split into four new wallets with four different assets, merged in GBP within 2 minutes.
  The collector's next payment, to a counterparty it already knew, ranks 17,189: we score the motif, not the wallet.

**Look-alike we leave alone:** `rnpfKt4h…`, the busiest wallet in the test week (316 payments, 247
counterparties, active every day). Extreme fan-in and fan-out, but a steady 20-minute business pace and no
pass-through: none of its payments reach our top 1% (best rank 1,975).

---

## 8. Limits and next steps

- The data is synthetic. A high score here proves nothing about real financial crime.
- **Hard relay cases are out of reach.** 200 suspicious relay transactions sit among 19,800 normal ones built from
  the same template. We tested about 15 signals (sender recurrence, burst overlap, onward chains, repeated paths,
  amount trends, timing, assets) at transaction and whole-scenario level: none beat AUC 0.57. Their wallets are
  ordinary wallets (about 92% of each one's activity is normal).
- 25 more hard labels are random by design. Together this caps any honest model at roughly AP 82.
- The test set has variations absent from training. TODO: what we did about it.
- Next: more labels for the hard cases, longer history than one week, account-level models.

---

## 9. AI disclosure

- TODO: tools and models, what each did (copy from `ATTRIBUTION.md`).
- We ran every result ourselves. All scores come from our code, not from the AI.
