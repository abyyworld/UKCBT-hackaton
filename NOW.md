# Who is working on what right now

**Both AIs: read this before starting anything, and update it when you start or finish.** If a task is claimed by
the other person, do not work on it. Pick an open task or ask your human. Pull, edit this file, push.

| Task | Owner | Status | Files |
|---|---|---|---|
| Local, episode and fingerprint features | Aby | done | `src/features/local.py`, `episodes.py`, `fingerprint.py` |
| Stage 2 group scoring, auxiliary targets, model settings | Aby | in progress | `src/stage2.py`, `src/cv.py` |
| `reason` column, `run.py`, final predictions, uploads, code ZIP | Aby | in progress | `src/predict.py`, `run.py`, `scripts/` |
| Relay hub features (hard cases) | Partner | done, dropped (no gain) | `explore/relay_chain.py` |
| Train vs test shift check | Partner | done | `explore/ADV_VAL.md` |
| LICENSE, ATTRIBUTION, SUBMISSION | Partner | done, update at the end | root files |
| **Medium-case features (slow motifs)** | Partner | open, next | `src/features/medium.py` |
| Slides | Partner | after medium | `report/` |

## Current best scores (out-of-fold AP, grouped 5-fold)

| Version | ALL | easy | medium | hard |
|---|---|---|---|---|
| local only, 2 seeds | 59.25 | | 6.91 | |
| local + episodes, 2 seeds | 58.18 | | 6.98 | |
| local + episodes + stage 2, 1 seed | 59.68 | 75.30 | 7.06 | 1.59 |

Leaderboard: v1 (local only) score TODO
