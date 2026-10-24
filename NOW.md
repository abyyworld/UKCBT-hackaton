# Who is working on what right now

**Both AIs: read this before starting anything, and update it when you start or finish.** If a task is claimed by
the other person, do not work on it. Pick an open task or ask your human. Pull, edit this file, push.

| Task | Owner | Status | Files |
|---|---|---|---|
| Local, episode and fingerprint features | Aby | done | `src/features/local.py`, `episodes.py`, `fingerprint.py` |
| Stage 2 group scoring: neighbour features, rounds, settings | Aby, second session | in progress | `src/stage2.py` |
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
| **Medium slow motifs (helper 1)** | Partner helper 1 | in progress | `src/features/slow.py`, `explore/SLOW.md` |
| **Pseudo-labelling / transductive check** | Partner helper 2 | done, dropped (+0.03 on fresh seeds) | `explore/pseudo.py`, `explore/PSEUDO.md` |
| **Medium wallet reuse (helper 2)** | Partner helper 2 | in progress, push by 13:30 UTC | `src/features/reuse.py`, `explore/REUSE.md` |
| **Medium value matching (helper 3)** | Partner helper 3 | in progress | `src/features/match.py`, `explore/MATCH.md` |
| Hard relay signal hunt (pass-through, cover, gap) | Partner helper 3 | done, dropped: ALL 60.01 vs 61.23, hard 0.92 vs 0.89 (seed 0). Scheme-level signal is real but needs scheme reconstruction, see HARD.md | `explore/HARD.md` |

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
