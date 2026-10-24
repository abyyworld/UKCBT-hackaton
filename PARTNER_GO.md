# Partner go-plan

## CURRENT DIRECTION (read this first, updated 13:05)

**Score:** our best leaderboard run is 68.6 (v16, a blend of v13 and v14). The leader is at 71.1, so we need about +2.5.
**Why we are behind:** the top 1% is already full of easy cases for every team. The points left are in the middle of the
ranking: medium cases (9.7 points available) and fast normal look-alikes. Hard relay cases hold 13.6 points but nobody
catches them yet.

**Who does what (check `NOW.md`, never duplicate):**
| Owner | Work |
|---|---|
| Aby (main session) | combines everything, uploads, `run.py`, final blend, v17 real-vs-look-alike run |
| Second session | stage 2 (`src/stage2.py`) |
| Hunter 1 | hard relay cases (`src/features/hardrelay.py`) |
| Hunter 2 | pushing normal hub traffic down (`src/features/hubs.py`) |
| Alternative method session | whole-scheme model from scratch for blending (`scripts/scheme_model.py`) |
| **Partner helpers 1 to 3** | **medium cases: slow motifs, value matching, wallet reuse (keep these, they are claimed)** |
| **Partner** | **slides, `report/EXAMPLES.md`, `ATTRIBUTION.md` (all AI sessions used), final reproducibility check** |

**Shared baseline for every feature test:** `python -m src.cv --modules local,episodes,isolation,patterns,shape`
(seed 0, stage 1): ALL 61.23, easy 80.75, medium 9.42. A new module goes in only if ALL rises by 0.5 or more, or
medium rises by 1 or more with ALL not dropping. Push only the module file (`src/features/<name>.py`) plus a NOW.md row
with the numbers. The main session then adds it to the final run.

**Timeline:** findings pushed by 14:45. Feature freeze 15:00. Final run 15:00 to 15:30 and upload by 15:30.
Slides final by 15:45 with the final numbers (ask Aby). Code ZIP and everything uploaded by 16:15.

**Rules reminder:** commits as plain messages, no AI trailers. No labels, family, difficulty, scenario_id or cohort_id
as features. No em dashes in anything humans read.

---

**For the partner:** open this repo in Claude Code (or any AI coding assistant) and send:
"Read PARTNER_GO.md and TEAM_RULES.md, then wait. When I type go, carry out the tasks in order."
Then type **go**.

---

## Instructions for the AI assistant

You are helping the second member of team Batman and Robin in the Decode & Detect hackathon (submissions close
16:30 London). Work through the tasks below **in order**. After each task, commit and push (rules below), then
report back to the user in one or two lines with the numbers you measured. Do not wait for approval between tasks
unless something fails.

### Ground rules (from TEAM_RULES.md, follow them strictly)
- Run `git pull --rebase` before every push. Commit only the files you own (listed per task).
- **Never edit** `src/cv.py`, `src/predict.py`, `src/load.py`, `src/schema.py`, `src/features/__init__.py`,
  `src/features/local.py`, `src/features/episodes.py`, `src/features/stage2.py` or `run.py`. Aby's session owns
  them. If you need a change there, write it as a suggestion in `explore/NOTES.md`.
- Never commit `data/`, `submissions/`, zips, `.venv/`, access codes or keys. The repo is public.
- Never use `label`, `scenario_id`, `cohort_id`, `family`, `difficulty` or the split name as features.
- A feature module stays only if out-of-fold AP from `python -m src.cv` goes up. Otherwise delete it.
- Human-facing text: plain words, no em dashes.

### Task 0: setup (10 min)
1. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
2. Unzip the two portal downloads so these paths exist:
   - `data/decode-and-detect-training-data/train/transactions.parquet` (also `validation/`, `README.md`)
   - `data/decode-and-detect-test-data/transactions.parquet`
3. Run `.venv/bin/python -m src.cv --tag baseline`. Expect about `ALL 58.6  easy 77.5  medium 5.8  hard 0.8`.
   Report the line to the user.

### Task 1: relay hub features, the hard cases (45 min). You own `src/features/relay.py`
Context: 200 hard suspicious transactions hide inside about 20,000 normal relay-hub transactions
(`aggregate_relay`, `distributed_relay` families in `train/labels.csv`). A suspicious collector receives from a
recurring set of about 7 senders in bursts of about 80 minutes, on several different days, all received as XRP, then
pays out hours later. Suspicious relay amounts are larger (median USD about 165 against 114 for normal ones).
Write `build(txs) -> pandas.DataFrame` indexed by `txs["tx"]`, numeric columns only, no labels. Use
`from src.load import load_split` to get `txs` (columns: tx, ts, sender, receiver, asset_in, asset_out, usd_in,
usd_out, n_legs, is_conversion). Ideas, keep what helps:
- Per receiver: split its receives into bursts (gap over 30 min starts a new burst). Burst size, burst duration, number
  of bursts in the week, number of distinct days with bursts.
- Sender recurrence: for this transaction's sender, how many other bursts of the same receiver it also appears in.
  Jaccard overlap of sender sets between the receiver's bursts.
- Asset consistency: share of the receiver's receives that arrive in the same asset as this one.
- Burst in/out balance: USD received in the burst against USD the receiver sends in the next 12 hours.
- This transaction's USD against the receiver's median and the burst median.
Measure with `.venv/bin/python -m src.cv --modules local,relay --tag relay`. Compare `ALL` and `hard` with the
baseline. Push `src/features/relay.py` only if `ALL` goes up. Report both lines to the user.

### Task 2: robustness check (30 min). You own `explore/adv_val.py` and `explore/ADV_VAL.md`
The test data has "batch variations absent from training". Train a LightGBM classifier to tell train rows from test
rows using the `local` features (load them with `from src.features import build_split`; `build_split("train",
["local"])` and `build_split("test", ["local"])` return `(txs, X)`). Report the AUC and the 15 features with the
largest importance. In `explore/ADV_VAL.md`, list the features that look like they shift (AUC per feature above 0.6)
so Aby can drop or rescale them. Push both files.

### Task 3: submission paperwork (30 min). You own `LICENSE`, `ATTRIBUTION.md`, `SUBMISSION.md`
- `LICENSE`: MIT, "Copyright (c) 2026 team Batman and Robin".
- `ATTRIBUTION.md`: what we did not write ourselves. Libraries from `requirements.txt` with their licences; the
  organiser data and organiser `payments.py` notes (we do not ship the data); AI assistance: name each AI
  coding assistant and model the team used, what it was used for, and that prompts are summarised in the report.
  The rules require this disclosure.
- `SUBMISSION.md`: how to reproduce. Python 3.13, `pip install -r requirements.txt`, where to unzip the data,
  the exact command (`python run.py`, which writes `submissions/final.csv`), and the hardware and time
  (about 2 minutes on a 4-core laptop). Leave a TODO line for the final out-of-fold and leaderboard scores.

### Task 4: slides skeleton (until 15:00). You own `report/`
Create `report/SLIDES.md`, fewer than 10 slides: problem, data, what makes laundering visible here (tempo,
first-time pairs, episodes), features, model and validation (grouped 5-fold out-of-fold AP), results (leave
placeholders), two or three flagged cases with transaction hashes, one look-alike we deliberately do not flag, limits
and next steps, AI disclosure. Ask the user for the latest scores at 15:00 and fill them in.

### Task 5: medium cases (the next big gain). You own `src/features/medium.py`
Check `NOW.md` first and mark this task "in progress". Medium suspicious cases (225 in train, AP only about 6) are
the same motifs as the easy ones but stretched over about 1.8 hours, with gaps of about 18 minutes, built almost
entirely from first-time wallet pairs (98% against 55% for normal payments). Normal look-alikes are faster
(about 10 minute gaps, about 1 hour span). Write `build(txs)` that finds slow motifs among first-time pairs only:
- fan-out: the sender's number of first-time recipients within 30m, 1h, 2h and 4h after this payment, and how even
  the gaps are;
- fan-in: the receiver's number of first-time senders within the same windows before this payment;
- chain: the receiver pays a first-time pair onward 5 to 60 minutes later, then that wallet does the same (depth);
- bipartite and stack: senders of first-time pairs that share two or more first-time recipients within 3 hours
  (complete small blocks of new pairs);
- these counts relative to what the same wallet does on other days of the week.
Compare `python -m src.cv --modules local,medium --tag medium` with `python -m src.cv --modules local --tag base`
(single-seed runs move by about 1 point, so only trust gains above that). Push `medium.py` only if ALL rises.
Write the numbers in `NOW.md`.

### When done or stuck
Write what you found in `explore/NOTES.md`, push, and tell the user.
