# Final steps checklist (anyone can finish from here)

Deadline: **16:30 London**. Aim to have everything uploaded by **16:15**.

## 1. Final predictions (Aby)
- Freeze the model at 15:00 London. `python run.py` writes `submissions/final.csv` (about 7 minutes from scratch,
  checked twice; v13 out-of-fold 63.81).
- Upload the CSV and note the leaderboard score in `NOW.md`.

## 2. Fill the TODOs (partner)
- `SUBMISSION.md` > Results: final out-of-fold AP, leaderboard score, commit hash (`git rev-parse --short HEAD`).
- `ATTRIBUTION.md`: the exact AI model names each of you used (shown in your Claude or other AI app), and Aby's tool.
- `report/SLIDES.md`: results table "Final" row; check the examples in `report/EXAMPLES.md` still match the final
  file (ranks can shift slightly; the hashes stay valid examples).
- `TEAM.md`: team name (Batman and Robin), captain, members, final commit.

## 3. Code ZIP (≤ 8 MiB, ≤ 64 MiB unzipped, ≤ 2,000 files)
- `bash scripts/make_zip.sh` writes `submissions/code.zip` (source, docs, requirements, licence; no data).
- Check it: `unzip -l submissions/code.zip` must show no `data/`, `.venv/`, `.parquet`, zips or keys.
- It must contain: `src/`, `run.py`, `requirements.txt`, `README.md`, `SUBMISSION.md`, `LICENSE`, `ATTRIBUTION.md`,
  `report/` (slides), `TEAM.md`, and the predictions CSV plus validation results if the portal asks for them.

## 4. Upload
- On https://hackathon.ukcbt.org upload **the code ZIP and `final.csv` together**. Confirm the portal shows a score.
- Then complete any supporting-material step the portal asks for.
