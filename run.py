"""One command from the raw data to the upload file.

    python run.py            # writes submissions/final.csv (the submitted rank blend, about 50 minutes)
    python run.py --single   # writes submissions/final.csv from the v17 model alone (about 10 minutes)

The submitted file is a rank average of differently built models. Each one is trained on train only, scores the test
week, and is written to submissions/<tag>.csv; src/blend.py then averages their ranks.
- v17: stage 1 LightGBM, a real-vs-look-alike model on motif-shaped payments, and stage 2 group scoring, on the
  local, episodes, isolation, patterns and shape features (scripts/realmotif_run.py).
- v28: the same recipe on within-week percentile features plus the slow closure features, without the features that
  shift between train and test (scripts/robust2_run.py with SPLIT_RANK=1).
- v17r: v17 without the amount-evenness features (explore/robust_run.py).
- v26: rank average of v17 and a test-week pseudo-labelling model cross-fitted on test (scripts/pseudo_adapt.py).
- scheme: a scheme-level model that scores candidate schemes and passes the score to their payments
  (scripts/scheme_model.py).
- v13cf: stage 1 plus a fold-consistent stage 2 with first-time-pair neighbour scores (src/stage2.py --extra ftp).
The last step adds a plain-English reason column (scripts/add_reasons.py); it does not change the scores.
Seeds are fixed, so a re-run gives the same file.
"""
import os
import subprocess
import sys

PY = sys.executable
SUB = "submissions"
RANK_SLOW = {"SPLIT_RANK": "1", "MODULES": "local,episodes,isolation,patterns,shape,slow"}
MEMBERS = ["v17", "v28", "v17r", "v26", "scheme", "v13cf"]
STEPS = [
    ({}, [PY, "scripts/realmotif_run.py", "--tag", "v17", "--seeds", "2"]),
    (RANK_SLOW, [PY, "scripts/robust2_run.py", "--tag", "v28", "--seeds", "2"]),
    ({}, [PY, "-m", "explore.robust_run", "--tag", "v17r", "--seeds", "2"]),
    ({}, [PY, "scripts/pseudo_adapt.py", "--base", "v17"]),
    ({}, [PY, "-m", "src.blend", "--out", "v26", f"{SUB}/v17.csv", f"{SUB}/pseudo.csv"]),
    ({}, [PY, "scripts/scheme_model.py"]),
    ({}, [PY, "-m", "src.stage2", "--tag", "v13cf", "--modules", "local,episodes,isolation,patterns,shape", "--sub", "none",
          "--seeds", "2", "--consistent", "--extra", "ftp", "--predict", "--testpred", "folds"]),
    ({}, [PY, "-m", "src.blend", "--out", "final"] + [f"{SUB}/{m}.csv" for m in MEMBERS]),
    ({}, [PY, "scripts/add_reasons.py", f"{SUB}/final.csv"]),
]
SINGLE = [({}, [PY, "scripts/realmotif_run.py", "--tag", "final", "--seeds", "2"])]

if __name__ == "__main__":
    for env, step in SINGLE if "--single" in sys.argv else STEPS:
        print(">>", " ".join(f"{k}={v}" for k, v in env.items()), " ".join(step[1:]), flush=True)
        subprocess.run(step, check=True, env={**os.environ, **env})
    print("done: submissions/final.csv")
