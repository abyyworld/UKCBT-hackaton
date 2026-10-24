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

- **Tempo:** money leaves soon after it arrives.
- **First-time pairs:** payments between wallets that have never dealt with each other.
- **Episodes:** bursts of related payments, repeated across several days with the same small set of senders.
- TODO: one chart (for example, time from receive to forward, suspicious vs normal).

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
| Baseline | 58.6 | 77.5 | 5.8 | 0.8 | TODO |
| TODO | | | | | |
| Final | TODO | | | | TODO |

---

## 7. Flagged for review: examples

- TODO: transaction hash 1, why it scored high (shape, timing, amounts).
- TODO: transaction hash 2.
- TODO: transaction hash 3.

**Look-alike we leave alone:** TODO: an exchange or payroll hub with huge fan-in or fan-out, and why its pattern
looks normal (steady, many unrelated customers, no quick pass-through).

---

## 8. Limits and next steps

- The data is synthetic. A high score here proves nothing about real financial crime.
- About 25 hard labels are deliberately random, so no model can find them.
- The test set has variations absent from training. TODO: what we did about it.
- Next: more labels for the hard cases, longer history than one week, account-level models.

---

## 9. AI disclosure

- TODO: tools and models, what each did (copy from `ATTRIBUTION.md`).
- We ran every result ourselves. All scores come from our code, not from the AI.
