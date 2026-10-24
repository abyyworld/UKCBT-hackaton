# Model settings and blends

Run: `python -m explore.models local` (same grouped 5-fold split as `src.cv`, seeds 0, 1, 2, `local` features).

| Model | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| LightGBM, current PARAMS | 59.53 | 60.41 | 56.86 | 58.93 |
| num_leaves 15, min_child 40 | 58.71 | 59.90 | 56.70 | 58.44 |
| learning_rate 0.015, 1400 trees | 59.55 | 60.73 | 57.11 | 59.13 |
| scale_pos_weight 5 | 59.65 | 60.59 | 57.26 | 59.17 |
| feature_fraction 0.4, lambda_l2 5 | 59.27 | 59.87 | 56.71 | 58.62 |
| ExtraTrees (500 trees, leaf 3, max_features 0.3) | 58.40 | 57.42 | 55.11 | 56.98 |
| Rank blend: LightGBM + ExtraTrees | 60.20 | 60.44 | 56.88 | 59.17 |
| **Rank blend: all five LightGBM + ExtraTrees** | 61.00 | 61.00 | 57.60 | **59.62** |

**Suggestion for Aby:** the full rank blend beats the current model on every seed (+0.7 on average). Small but
steady and cheap at prediction time. `scale_pos_weight=5` or the slower learning rate alone each add about +0.2.
Settings are not the main lever; features are.
