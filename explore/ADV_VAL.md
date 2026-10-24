# Train vs test shift (adversarial validation)

Run: `python -m explore.adv_val` (LightGBM, 5 folds, seed 0, `local` features only).

| Folds | Train vs test AUC |
|---|---|
| Random | 0.990 |
| Grouped by sender | 0.845 |

**What this means.** No single feature shifts: the largest single-feature AUC is 0.548 (`spread`), then 0.529
(`log_usd`). Nothing comes near 0.6. The high combined AUC comes from the per-wallet weekly counts (`s_out_week`,
`r_in_week`, `s_in_week`, `s_distinct_dst`, `r_distinct_src`, `r_out_week`, about 85% of the gain). Wallets never
appear in both splits, so together these counts act as a wallet fingerprint, and the classifier learns which wallet
belongs to which split rather than a real change in behaviour. Grouping folds by sender removes part of this (the
receiver fingerprint still leaks), and the AUC drops from 0.99 to 0.85.

**Suggestion for Aby.** No feature needs dropping or rescaling for shift. The week counts are the features to watch:
if the leaderboard score falls well below out-of-fold AP, try them as ranks within the split, or remove them and
compare out-of-fold AP.

Top 15 features by gain:

| Feature | Share of gain | AUC alone |
|---|---|---|
| s_out_week | 0.193 | 0.501 |
| r_in_week | 0.178 | 0.505 |
| s_in_week | 0.144 | 0.506 |
| s_distinct_dst | 0.123 | 0.514 |
| r_distinct_src | 0.116 | 0.503 |
| r_out_week | 0.087 | 0.508 |
| log_usd | 0.029 | 0.529 |
| pair_n | 0.021 | 0.507 |
| amt_z_sender | 0.017 | 0.503 |
| amt_z_receiver | 0.013 | 0.502 |
| spread | 0.011 | 0.548 |
| out_is_xrp | 0.010 | 0.521 |
| r_in_3600 | 0.008 | 0.509 |
| novel_s_86400 | 0.007 | 0.511 |
| novel_both_21600 | 0.007 | 0.506 |
