# Team rules (Batman and Robin)

Read this before your first push. These rules keep us fast, keep the entry valid, and stop us from overwriting each other.

## Git
- Everyone works on `main`. Run `git pull --rebase` before every push.
- Partner owns: `explore/`, `report/`, `src/features/relay.py`, `LICENSE`, `ATTRIBUTION.md`, `SUBMISSION.md`. Aby owns everything else in `src/` and `run.py`. Suggest changes to files you do not own in `explore/NOTES.md` instead of editing them.
- Partner: start with `PARTNER_GO.md`.
- Never commit `data/`, any zip, `.venv/`, the portal access code, passwords or API keys. The repo is public.
- Small commits with clear messages.

## Competition rules we must follow
- Use only the supplied training and test files. No outside labels, no test answers, no organiser-only files, no other teams' work.
- Never use which split a row comes from as a feature. Never use `scenario_id`, `cohort_id`, `family` or `difficulty` as features. They are only for grouping folds and analysis.
- Do not probe the leaderboard to learn individual test labels. Use uploads only to compare honest model versions.
- Disclose every AI tool, model and outside service you use. Write it in `report/` so it goes into the final ZIP and slides.
- Never call an address criminal. Say "flagged for review" and show the evidence.

## How ideas get in
1. Write the idea as one line, for example: "collector receives from the same 5 senders in two bursts on different days".
2. Prototype it in `explore/` using `from src.load import load_split, load_labels`.
3. It goes into the model only if out-of-fold AP from `python -m src.cv` goes up. If it does not help, it is dropped, however good it sounds.

## Writing for humans
- Slides, README and reports: plain words, short sentences, no em dashes.

## Partner focus
- The hard cases: suspicious payments hidden inside relay hubs (`aggregate_relay`, `distributed_relay` in `train/labels.csv`). Find what separates the 200 suspicious ones from the 20,000 normal ones.
- The slides (fewer than 10): problem, data, features, model, validation, results, limits, AI disclosure, two or three flagged cases with transaction hashes, one look-alike we deliberately do not flag. Ready by 15:30.
