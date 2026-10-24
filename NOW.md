# Who is working on what right now

**Both AIs: read this before starting anything, and update it when you start or finish.** If a task is claimed by
the other person, do not work on it. Pick an open task or ask your human. Pull, edit this file, push.

| Task | Owner | Status | Files |
|---|---|---|---|
| Local, episode and fingerprint features | Aby | done | `src/features/local.py`, `episodes.py`, `fingerprint.py` |
| Stage 2 group scoring: neighbour features, rounds, settings | Aby, second session | in progress | `src/stage2.py` |
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
| **Pseudo-labelling / transductive check** | Partner helper 2 | in progress | `explore/pseudo.py`, `explore/PSEUDO.md` |

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

LB is about OOF + 5.7, but differences under about 1 point are noise. Leader: 68.7 (10% of medium in their top 1%, we have 0%).
**v13 = new base for everyone: `--modules local,episodes,isolation,patterns,shape`, stage 2, 2 seeds.
OOF stage 1 61.23, stage 2 63.81 (easy 79.2, medium 10.0, hard 2.1). `python run.py` reproduces it.**
Uploaded for LB: pending. Next: more pattern windows (v14).
