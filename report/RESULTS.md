# Results log (final afternoon)

Every leaderboard upload of the afternoon, every method tried, and what we learned. Scores are 100 x average
precision on the full test set. OOF = out-of-fold AP on train (grouped folds). All scripts are in this repo.

## Final submission: 70.3

Rank average of six differently built models (`python run.py`, about 50 minutes):
v17, v28, v17r, v26, scheme, v13cf (see `SUBMISSION.md` for what each one is). The final file adds a plain-English
`reason` column (`scripts/add_reasons.py`); scores are unchanged.

## Leaderboard uploads

| Score | File | What it is |
|---|---|---|
| **70.3** | blend12 = final | blend6 + v13cf (fold-consistent stage 2 with first-time-pair neighbour scores) |
| 70.3 | blend13 | same, v13cf counted twice |
| 70.2 | blend6 | v17 + v28 + v17r + v26 + scheme |
| 70.1 | blend6 + fam | adds the per-method experts |
| 70.0 | blend6, 50 rows swapped from ltr | 10% of medium in the top 1%, but lower AP |
| 69.9 | blend8 | blend6 with scheme counted twice |
| 69.6 | blend4, v17 + v28 | blend4 = v17 + v28 + v17r + v26 |
| 69.4 | blend3, blend6 + ltr | |
| 69.2 | v17 | single model |
| 69.2 | blend6, 100 rows swapped from ltr | 10% of medium in the top 1% |
| 69.0 | v17r | v17 without amount-evenness features |
| 68.9 | v13c | fold-consistent stage 2 without subset models |
| 68.8 | v28, v26, blend5 | blend5 = blend3 + GNN at half weight |
| 68.5 | blend7 | blend6 + ExtraTrees at half weight |
| 67.9 | blend12, 40 relay rows lifted | hard-case score from `scripts/hard_odd.py`; Hard stayed at 0% |
| 60.9 | same, lifted to the very top | |
| 60.9 / 53.4 / 37.7 | blend12 with 40 / 100 / 366 test-week odd relays at the top | `scripts/test_odd.py`; Hard stayed at 0%, all were normal |
| 70.3 / 70.3 | blend12 with v28 counted twice / three times | amount scale shift check; no change |
| 54.6 | GNN alone | `scripts/gnn_model.py` |
| 53.8 | learning to rank alone | `scripts/ltr_run.py`; catches 10% of medium but ranks easy cases worse |

## Methods tried this afternoon (OOF, v17 = 67.47 is slightly optimistic)

| Method | Script | OOF | Verdict |
|---|---|---|---|
| Graph diffusion of v17 scores | `scripts/diffuse_run.py` | 67.71 | near copy of v17 |
| Cascade re-ranker on the top K | `scripts/cascade_run.py` | 66.50 | worse |
| Per-method experts + combiner | `scripts/family_experts.py` | 64.80 | 70.1 in the blend |
| v13cf stage 2 | `src/stage2.py --consistent --extra ftp` | 63.02 | in the final (+0.1) |
| Learning to rank (lambdarank, 2 h queries) | `scripts/ltr_run.py` | 62.01 | beats binary LightGBM on the same inputs (60.13) |
| Scheme-level model | `scripts/scheme_model.py` | 60.39 | the most useful blend member (+0.6) |
| ExtraTrees | `scripts/extratrees_run.py` | about 60 | hurts the blend |
| CatBoost | `scripts/catboost_run.py` | 59.71 | not uploaded |
| IBM Graph Feature Preprocessor recipe from scratch | `scripts/gfp_run.py` | 54.09 | worse than our features |
| GNN (edge message passing) | `scripts/gnn_model.py` | 47.39 | hurts the blend |

## What we learned

- **Blending differently built models is what moved the leaderboard** (69.2 to 70.3). Models that are weak alone
  but built differently (scheme 60 OOF) helped more than strong but similar ones (v17r, ExtraTrees).
- **Medium:** forcing 10% of medium into the top 1% is possible (learning to rank does it), but on our data the rows
  it brings in come with too many normal look-alikes, so AP falls (`scripts/medium_quota.py`: on train, a 10% quota
  costs about 6 AP).
- **Hard relays:** two week-long flow hunts (`scripts/hard_flow.py`, `scripts/hard_odd.py`) and a test-week
  odd-one-out search (`scripts/test_odd.py`) found no usable signal (best within-relay AUC 0.58 to 0.69, base rate 1%).
  The leader catches 20% of hard; we never found how.
- **Test-only batch scan:** every raw field (currencies, fees, flags, memos, tags, paths, sequences) matches train once
  token issuers are normalised. The one real shift is amount scale: test IOU amounts are lower (GBP median 33.7 vs
  56.5 in train), which favours within-week rank features (v28).
- **Validation leak in stage 2:** plain OOF of stacked models was optimistic (v17 67.47 vs 63.49 fold-consistent),
  which is why OOF gains stopped transferring to the leaderboard. `--consistent` in `src/stage2.py` fixes it.
