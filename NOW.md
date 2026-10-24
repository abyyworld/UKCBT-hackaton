# Who is working on what right now

## HARD RULE FROM 13:50 (overrides everything below)

**All model runs, experiments, CV comparisons and uploads happen ONLY on Aby's side** (Aby's main session and its four
helper sessions). Two machines computing the same thing wastes the last hours.

**Partner side does NOT run models any more.** Partner and partner helpers work only on:
1. `report/SLIDES.md` and `report/EXAMPLES.md` (final numbers at 15:30 from Aby).
2. `ATTRIBUTION.md`, `SUBMISSION.md`, `TEAM.md`, `report/FINAL_STEPS.md`.
3. The final reproducibility check of `run.py` after Aby's final run (about 15:30).
4. Ideas: write each as one line under "Idea queue" in NOW.md. Aby's side decides and runs them. Do not run them.

Already running on Aby's side, do not repeat: v22 (robust + slow), scheme model, test-week pseudo-labelling
(`scripts/pseudo_adapt.py`), importance reweighting (`scripts/reweight_run.py`), shift audit (`explore/SHIFT.md`).


**Both AIs: read this before starting anything, and update it when you start or finish.** If a task is claimed by
the other person, do not work on it. Pick an open task or ask your human. Pull, edit this file, push.

| Task | Owner | Status | Files |
|---|---|---|---|
| Local, episode and fingerprint features | Aby | done | `src/features/local.py`, `episodes.py`, `fingerprint.py` |
| Stage 2 group scoring: neighbour features, rounds, settings | Aby, second session | in progress. **Nested holdout warning:** outer grouped 5-fold acting as a fake test set, v17 pipeline (local,episodes,isolation,patterns,shape + realmotif, 2 seeds) run inside each outer fold: stage 2 mean per-fold AP 61.00 vs stage 1 alone 62.72 (pooled 57.75 vs 60.87). Test scoring from fold models (`--testpred folds`) is no better (60.99). Likely cause: normal payments on planted wallets are in other folds, so their OOF stage-1 scores leaked the planted labels, and stage 2 learns to trust leaked neighbour scores. Fix in test: `--consistent` (each row's neighbours scored by the fold model that never saw its fold). Suggest keeping one pure stage 1 upload on the v13 modules as a check | `src/stage2.py` |
| Hard relay cases hunter | Aby, hunter session 1 | in progress, report by 14:45 | `src/features/hardrelay.py`, `explore/HARD.md` |
| Normal hub down-ranking (to lift medium) | Aby, hunter session 2 | in progress, report by 14:45 | `src/features/hubs.py`, `explore/MEDIUM2.md` |
| Alternative method: whole-scheme classifier (rebuild schemes, score each scheme, map to payments) | Aby, alt session | in progress, due 14:40 | `scripts/scheme_model.py` |
| Final blend of best variants (v13, v14, v15, stage 2 real-vs-look-alike) | Aby | in progress | `src/blend.py`, `run.py` |
| Scheme isolation features (wallets do nothing else during a scheme) | Aby | in progress | `src/features/local.py` |
| `reason` column, `run.py`, final predictions, uploads, code ZIP | Aby | in progress | `src/predict.py`, `run.py`, `scripts/` |
| Relay hub features (hard cases) | Partner | done, dropped (no gain) | `explore/relay_chain.py` |
| Train vs test shift check | Partner | done | `explore/ADV_VAL.md` |
| LICENSE, ATTRIBUTION, SUBMISSION | Partner | done, update at the end | root files |
| Medium-case features (slow motifs) | Partner | done, dropped (no gain) | `src/features/medium.py` |
| Model settings and model blends | Partner | done: rank blend +0.7, see explore/MODELS.md | `explore/models.py`, `explore/MODELS.md` |
| Slides | Partner | after medium | `report/` |
| **Easy look-alike separation (motif shape)** | Partner helper 1 | done, kept v3: +1.11 over local, +1.74 over local+episodes+isolation+patterns (3 seeds); v4 dropped. Add `shape` to modules. See SHAPE.md | `src/features/shape.py`, `explore/SHAPE.md` |
| **Medium slow motifs (helper 1)** | Partner helper 1 | done, kept v2 closure. `python -m src.cv` seed 0 here: base 59.50 (medium 8.44), +slow 60.30 (medium 7.55), +0.80 ALL. 3 seeds: +0.55 ALL, medium flat about 8. Add `slow` to modules. See SLOW.md | `src/features/slow.py`, `explore/SLOW.md` |
| **Pseudo-labelling / transductive check** | Partner helper 2 | done, dropped (+0.03 on fresh seeds) | `explore/pseudo.py`, `explore/PSEUDO.md` |
| **Medium wallet reuse (helper 2)** | Partner helper 2 | done, dropped (lean +0.33 ALL, +0.27 medium over 5 seeds), see REUSE.md | `src/features/reuse.py`, `explore/REUSE.md` |
| **Medium value matching (helper 3)** | Partner helper 3 | done, dropped: best ALL +0.09, medium -0.15 (3 seeds). See MATCH.md | `src/features/match.py`, `explore/MATCH.md` |
| Hard relay signal hunt (pass-through, cover, gap) | Partner helper 3 | done, dropped: ALL 60.01 vs 61.23, hard 0.92 vs 0.89 (seed 0). Scheme-level signal is real but needs scheme reconstruction, see HARD.md | `explore/HARD.md` |
| Scheme-level model (alternative for rank blend) | Aby, scheme session | done: OOF ALL 60.39, easy 82.79, medium 4.03, hard 0.67 with 13 scales (one-hop schemes added) and no amount-evenness features (evenness shifts between train and test). Runs in about 3 min, writes data/oof/scheme.parquet and submissions/scheme.csv | `scripts/scheme_model.py` |

## Current best scores (out-of-fold AP, grouped 5-fold)

| Version | ALL | easy | medium | hard |
|---|---|---|---|---|
| local only, 2 seeds | 59.25 | | 6.91 | |
| local + episodes, 2 seeds | 58.18 | | 6.98 | |
| local + episodes + stage 2, 1 seed | 59.68 | 75.30 | 7.06 | 1.59 |

## Leaderboard

| Run | What | LB | Top 1% precision |
|---|---|---|---|
| 1 | v3 blend (stage 1 + stage 2, 50/50) | 66.5 | 94% |
| 2 | v2 stage 2, local+episodes, 3 seeds (OOF 61.63) | **67.3** | 93% |

| 3 | v7 stage 2 + local isolation features (OOF 61.94) | about 67.5 | 93% |
| 4 | v9 blend v2 + v7 | **67.6** | 93% |
| 5 | v8 stage 2 + episode isolation (OOF 62.35) and v11 blend v2+v7+v8 | 67.5 and 67.1 | 92 to 93% |

| 6 | v13 stage 2 + patterns + shape (OOF 63.81) | 68.5 | 92% |
| 7 | v16 blend v13 + v14 | **68.6** | 92% |

| 8 | **v17: real-vs-look-alike motif model + stage 2 on v13 features (OOF 67.47, medium 17.8)** | **69.1** | 93% |

LB is about OOF + 5.7, but differences under about 1 point are noise.
**Warning:** OOF is getting optimistic for stacked models (OOF to LB gap: v2 +5.7, v13 +4.7, v17 only +1.6). Prefer simple, robust gains; the leaderboard decides.
**New base from 13:12: v17 (`python run.py`, which calls `scripts/realmotif_run.py`).** Leader: 68.7 (10% of medium in their top 1%, we have 0%).
**v13 = new base for everyone: `--modules local,episodes,isolation,patterns,shape`, stage 2, 2 seeds.
OOF stage 1 61.23, stage 2 63.81 (easy 79.2, medium 10.0, hard 2.1). `python run.py` reproduces it.**
Uploaded for LB: pending. Next: more pattern windows (v14).

## Idea queue (partner writes one line each, Aby's side runs them)
- Medium band demotion: medium sits at ranks 1,000 to 1,600 among about 1,800 normals (1,119 relay, 405 slow look-alikes); scale scores x0.3 in ranks 800 to 3,000 for rows a relay-membership model (family as target only) calls relay. Oracle on v17 OOF: +1.6 ALL, medium 17.8 to 20.2; realistic +0.3 to +0.8. Cheap: a post-processing step on existing scores.
- Final upload as a rank blend of the three best honest LB runs (v17 69.2, v17r 69.0, v16 68.6): every blend so far beat or matched its parts, and v17/v17r differ on 56 of the top 401.
- Final run with 5 seeds instead of 2 for stage 1 and the real-vs-look-alike model: lower prediction variance, typically +0.2 to +0.5, costs only run time.
- Feed `slow` closure (clo14400) into stage 2 as a collective input (mean over linked payments) instead of a plain feature: it fires on 39% of medium vs 1.8% of normal but trees do not use it at stage 1.

