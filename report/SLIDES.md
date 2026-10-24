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

## 4. Features (five groups, all computed inside one week, no labels)

- **Local:** amount in USD, conversions, legs, how busy sender and receiver are over windows from 2 minutes to a
  day, and whether this is the first payment between the two wallets.
- **Episodes:** payments linked through a shared wallet within a short time form an episode; its size, span, step
  gaps and amounts describe the whole motif, not one payment. Tight scales for fast motifs, wider scales (first-time
  pairs only) for slow ones.
- **Scheme isolation:** one actor controls a scheme's wallets, so during the scheme they make no other payments.
  Normal look-alike wallets keep their regular relationships going.
- **Patterns among first-time pairs:** cycles back to the sender, how far money travels, parallel two-hop
  scatter-gather paths, and dense blocks of new pairs, over 2 and 6 hours (the slow medium schemes).
- **Motif shape:** inside each group, how much value pass-through wallets keep, how regular the hop delays are,
  how amounts change hop to hop. Laundering passes money on almost whole (98 to 100%).

---

## 5. Model and validation

- **Stage 1:** LightGBM on all features, averaged over 2 seeds.
- **Stage 2, collective scoring:** each payment also gets the stage 1 scores of the payments linked to it (same
  episode, same wallets nearby in time), and a second model learns from those. Missing steps of a scheme get pulled
  up by the rest of it.
- **Real vs look-alike motif model:** a third LightGBM trained only on rows from the motif scenarios (the label
  file's family column is used only to choose these training rows, never as an input). It learns what separates a
  planted motif from its normal look-alike, and its scores feed stage 2. This took out-of-fold AP from 63.8 to 67.5
  and medium from 10 to 18.
- **Validation:** grouped 5-fold out-of-fold AP. Every planted scenario stays inside one fold, so a model never
  sees part of a scheme it is scored on. Single-seed results move by about 3 points, so we only kept ideas that
  helped on several seeds.
- **Robustness:** no single feature shifts between train and test (largest single-feature AUC 0.55).
- No labels, scenario IDs, families, difficulty or split names are used as features.

---

## 6. Results

| Version | Out-of-fold AP | Easy | Medium | Hard | Leaderboard |
|---|---|---|---|---|---|
| Local features only (baseline) | 58.6 | 77.5 | 5.8 | 0.8 | |
| Local + episodes, stage 2 group scoring | 61.6 | | | | 67.3 |
| + scheme isolation | 61.9 | | | | 67.6 (blend) |
| v13: + first-time-pair patterns + motif shape, stage 2 | 63.8 | 79.2 | 10.0 | 2.1 | TODO |
| v17: + real-vs-look-alike motif model feeding stage 2 | 67.5 | | 17.8 | | TODO |
| Final | TODO | | | | TODO |

- The leaderboard runs about 5.7 points above our out-of-fold AP; differences under 1 point are noise.
- Ideas tested and dropped because they did not raise out-of-fold AP: relay hub features, slow-motif window counts,
  closed squares, a medium-only model, pseudo-labelling, wider pattern windows (see `explore/NOTES.md`).

---|---|---|---|---|---|
| Baseline | 58.6 | 77.5 | 5.8 | 0.8 | TODO |
| TODO | | | | | |
| v17: + real-vs-look-alike motif model feeding stage 2 | 67.5 | | 17.8 | | TODO |
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
