# Pseudo-labelling check (transductive): does not help

Question: train on train, score the unlabelled test rows, add the most confident ones back as pseudo-labels, retrain.
Does that lift AP?

## How we measured it
Inside grouped 5-fold CV (`StratifiedGroupKFold` on the scenario groups from `src/cv.py`). In each fold the held-out
fold plays the unlabelled test set: fit on the training folds, score held-out, take its top k as pseudo-positives
(and bottom m as pseudo-negatives) with weight w, refit on training folds plus pseudo rows, rescore held-out.
Held-out labels are used only for the final AP. Same LightGBM settings and seed for plain and pseudo models.

A held-out fold has about 8,000 rows and 250 positives. The test set has 40,101 rows. So k=20 / 60 / 100 / 150 in a
fold is about k=100 / 300 / 500 / 750 on test. m=4000 in a fold is about the bottom 20,000 on test.

Script: `python -m explore.pseudo --modules local --ks 100 --ms 0 --weights 1 --rounds 2`

## Results (AP x100, gain over plain on the same seed and folds)

### local, seeds 0, 1, 2 (mean)

| k (fold) | m | w | rounds | ALL | gain ALL | easy | medium | hard |
|---|---|---|---|---|---|---|---|---|
| plain | | | | 58.93 | | 77.24 | 7.37 | 0.82 |
| 20 | 0 | 0.5 | 1 | 59.01 | +0.08 | 77.35 | 7.64 | 0.81 |
| 20 | 0 | 1 | 1 | 58.92 | -0.02 | 77.29 | 7.50 | 0.81 |
| 20 | 4000 | 0.5 | 1 | 59.01 | +0.08 | 77.41 | 7.47 | 0.80 |
| 20 | 4000 | 1 | 1 | 58.90 | -0.04 | 77.25 | 7.23 | 0.80 |
| 60 | 0 | 0.5 | 1 | 59.17 | +0.24 | 77.63 | 7.43 | 0.80 |
| 60 | 0 | 1 | 1 | 59.34 | +0.41 | 77.95 | 7.85 | 0.79 |
| 60 | 4000 | 0.5 | 1 | 59.27 | +0.34 | 77.69 | 7.94 | 0.81 |
| 60 | 4000 | 1 | 1 | 59.32 | +0.39 | 77.89 | 7.64 | 0.81 |
| 60 | 0 | 1 | 2 | 59.28 | +0.35 | 77.83 | 7.81 | 0.80 |
| 100 | 0 | 1 | 1 | 59.80 | +0.86 | 78.54 | 8.16 | 0.80 |
| **100** | **0** | **1** | **2** | **60.09** | **+1.16** | 78.90 | 8.10 | 0.82 |
| 150 | 0 | 1 | 1 | 59.82 | +0.88 | 78.56 | 8.12 | 0.81 |
| 150 | 0 | 1 | 2 | 59.79 | +0.86 | 78.48 | 8.41 | 0.80 |

Per seed for the best row (k=100, 2 rounds): +0.73, +1.05, +1.69.

### Confirmation on fresh seeds 3, 4 (local, k=100, m=0, w=1, 2 rounds)

| seed | plain | pseudo | gain |
|---|---|---|---|
| 3 | 59.04 | 59.40 | +0.35 |
| 4 | 59.25 | 58.95 | -0.30 |
| mean | 59.15 | 59.17 | **+0.03** |

### local + episodes, seeds 0, 1, 2 (k=100, m=0, w=1)

| rounds | ALL | gain ALL | medium |
|---|---|---|---|
| plain | 57.70 | | 7.45 |
| 1 | 57.95 | +0.25 | 7.33 |
| 2 | 58.14 | +0.44 | 7.67 |

## Verdict
Does not help. The +1.16 on seeds 0 to 2 was the best of 14 settings picked on those same seeds. On two fresh seeds the
same setting gains +0.03. With local + episodes it gains +0.44, with one seed at +0.00. Seed-to-seed noise in ALL AP
is about 1.5 points, so none of these gains is real. Pseudo-negatives never helped. Only weak hint: medium went up on
most seeds with k=100 (about +0.5 to +1), but not reliably. Not recommended for the final model.

`pseudo_predict(X, y, X_test)` in `explore/pseudo.py` is a drop-in for `src.predict.fit_predict` if anyone wants to
try it on the leaderboard anyway (top 1.25% of test as pseudo-positives, 2 rounds).
