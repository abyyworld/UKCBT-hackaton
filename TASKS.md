# Who does what

## Aby (with AI assistant): the pipeline
- Load and join the data, build the features, train the model and validate it.
- Write `predictions.csv` and check it against the submission rules.
- Get a baseline submission onto the leaderboard first, then improve it.
- Package the code ZIP: run command, seeds and dependencies.

## Aby: admin and the portal
- Download the data package from the portal and pass it to the pipeline (**do not commit it**, the repo is public).
- Confirm the team name, captain and members, and keep the submission code private.
- Upload each new `predictions.csv` and log the score below.

## Partner: patterns and the report
- Use your own AI to explore `labels.csv` against the transactions. What do suspicious flows look like (shapes, timing,
  amounts, conversions)? Focus on the **hard** ones.
- Send findings as one-line feature ideas, e.g. "collector forwards >90% within 10 min".
- Push your code to `explore/` and your slides to `report/`, so we never edit the same files.
- **Every idea gets tested against local AP.** It stays only if it raises the score without leaking labels.
  Anything else is fixed or dropped.
- Start the slides early (fewer than 10): problem, data, features, model, validation, results, limitations, AI disclosure.

## Timeline
| By | What |
|---|---|
| Data in + 30 min | Baseline on the leaderboard |
| 13:00 | Model v1 with pattern features, submitted |
| 15:00 | Feature freeze |
| 15:30 | Final predictions submitted |
| 16:15 | ZIP and slides uploaded (15-minute buffer) |

## Git
Everyone works on `main`. Pull before you push. Never commit `data/`, access codes or credentials.

## Leaderboard log
| Time | Version | Local AP | Leaderboard |
|---|---|---|---|
