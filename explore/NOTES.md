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

## Task 5: medium-case features (dropped)
`src/features/medium.py`: first-time-pair fan-out and fan-in over 30 min to 4 h, gap evenness, onward chains of new
pairs (5 to 60 min), small complete blocks of new pairs, burst against the wallet's own week, gap to the nearest
other new-pair payment on the same wallets, and the median step inside groups of linked new-pair payments.
Five seeds: **local + medium mean ALL 58.69** against local about 58.5. Medium AP moved from about 7.2 to 6.7.
No real gain, so it was deleted.

What we learned (scenario medians in train):

| | Easy suspicious | Normal look-alike | Medium suspicious |
|---|---|---|---|
| Step gap | 1 min | 10 min | 18 min |
| Motif span | 5 min | 60 min | 110 min |

- Medium motifs run on busy wallets (about 110 payments a week each) and 55% of normal payments are also new pairs,
  so in any 1 to 2 hour window a normal busy wallet already has 3 or more new-pair payments. Window counts drown.
- Linking new-pair payments through shared wallets percolates: at 30 min links the groups merge into thousands.
  The median step inside the group is the best single medium signal (AUC 0.84 against normal), but its AP is only 2%.
- Single-seed out-of-fold AP moves by about 3 points between seeds (57 to 61). Only trust gains above 1 point on
  three or more seeds.

## Closed squares of first-time pairs, and a medium specialist (both dropped)
- `explore/squares.py`: a first-time-pair payment s -> r closes a square when s -> r2, s2 -> r and s2 -> r2 are
  also first-time pairs within +-W. With W = 1 h: exactly 3 squares is 21 medium against 3 normal, 2 to 3 squares
  is 33 medium against 61 normal (35% precision against a 0.6% base). With W = 3 h normal payments with squares rise
  from about 340 to about 2,300, so short windows are much sharper.
- As a module on top of `local,patterns` (3 seeds): ALL 59.24 vs 59.24, medium 9.09 vs 8.07. No ALL gain, deleted.
  **Suggestion for Aby:** if you want the medium gain, add 1800 and 3600 to `WINDOWS` in `patterns.py`
  (block_density covers the same idea).
- `explore/medium_specialist.py`: a second LightGBM trained on medium positives only, blended into the main score
  (sum, max, rank max). Every blend is worse than the main model alone (best: max 59.16 vs 59.24). The medium cases
  are not separable with enough confidence to beat easy cases at the top of the ranking.

## Pseudo-labelling (dropped)
See `explore/PSEUDO.md`. Best setting looked like +1.16 ALL on seeds 0 to 2 (local), but only +0.03 on fresh seeds
3 and 4, and +0.44 with local + episodes. Noise, so not used.

## Suggestion for Aby: code ZIP contents
`scripts/make_zip.sh` currently packs `src run.py requirements.txt README.md SUBMISSION.md LICENSE ATTRIBUTION.md
TEAM_RULES.md scripts`. The rules also want the slides or report, team details and the predictions with validation
results. Suggest adding `report TEAM.md explore/*.md` and `submissions/final.csv` (3 MB, fits the 8 MiB limit).

## Group-level classifier (for the scheme-model session; no gain as a blend)
`explore/groups.py`: first-time-pair payments linked through a shared wallet within T, one row per group (size,
span, gaps, assets in and out, conversion and XRP share, amounts, wallet busyness, value kept through pass-through
wallets, end-to-end conservation, max in and out degree). With T = 300 s each easy suspicious scenario lands in one
group (93% of its payments, 95% pure); slower look-alikes split into small pieces. Group label = majority label,
folds grouped by scenario. Group AUC 0.72 overall, **0.97 for groups of 3 or more payments**. But blending the group
score into v13 out-of-fold (sum, product, rank mix) never beats v13 (best 63.40 vs 63.81): the stage 1 and stage 2
features already carry the same information. Headroom check on v13 out-of-fold: pushing the normal look-alike
scenario payments (about 1,000 rows) to the bottom would lift AP from 63.8 to 78.6; perfect medium alone to 76.9.

## Three more checks on v13 out-of-fold (all dead ends)
- **Fund provenance for hard relays:** share of a relay payment's upstream inflows (sender, 6 h to 7 days before) or
  downstream outflows that are easy or medium suspicious payments: AUC 0.46 to 0.51 at every horizon. The relay money
  does not come from or go to other planted schemes.
- **Where the top false positives are:** in v13's top 400, 83 of 94 negatives are look-alike scenarios that are as
  fast as laundering (whole motif in 20 minutes or less); 9 are 20 to 60 minutes; 2 are background. Pushing every
  slower look-alike to the bottom would only reach 67.7 from 63.8; the fast look-alikes hold the rest.
- **Medium schemes are not isolated:** during a medium scheme its wallets make 11.1 other payments on average against
  11.3 expected from their weekly rate (easy: 0.6 vs 0.6, so isolation is not a medium signal either).
- **Demoting all relay traffic** (oracle, true family known): best +1.0 (scores x0.3). Not worth a model.

## v17 on the leaderboard: 69.2 (+0.6) while out-of-fold rose +3.7. Amounts look like the test variation.
Fast first-time-pair groups (4+ payments, median step 90 s or less, T = 300 s linking) have the same size, span,
gaps, wallet counts, assets, conversions and value kept in train and test. What differs is how even the amounts are:
coefficient of variation of amounts in the group, lower quartile / median / upper quartile 0.45 / 0.78 / 1.12 in
train vs 0.64 / 1.01 / 1.30 in test. In train, suspicious fast groups are even (median 0.45) and normal fast groups
uneven (0.91). So test laundering probably uses deliberately uneven amounts, and features that reward even amounts
would push it down. `explore/robust_run.py` runs v17 unchanged except without the amount-evenness features
(episode usd_std and rel_usd, shape hop ratios and kept_std, local amt_z). Its out-of-fold will be a little lower by
construction; only the leaderboard can tell whether it transfers better. Suggest one upload of it and one of a rank
blend with v17.

## Baseline numbers: 59.50 vs 61.23 is not a bug
`python -m src.cv --modules local,episodes,isolation,patterns,shape` (one seed) gives ALL 59.50 on any fresh
container. `run.py` / `scripts/realmotif_run.py` report "stage1 ALL 61.23" because they average the out-of-fold
scores of two seeds before computing AP. Both are reproducible; compare like with like.
- **v17r result:** out-of-fold stage 1 60.08 (v17 61.23), stage 2 65.21 (v17 67.47), medium 14.1 (v17 17.8). On test
  its ranking correlates 0.93 with v17; 56 of the top 401 differ, and the ones it brings in come from groups with
  uneven amounts (median CV 0.87 vs 0.71 for the ones it drops). Files: `python -m
  explore.robust_run --tag v17r --seeds 2`, then `python -m src.blend --out v17_v17r_blend submissions/final.csv
  submissions/v17r.csv`. Leaderboard: v17r 69.0 (v17 69.2), so dropping amount-evenness alone did not transfer better.
