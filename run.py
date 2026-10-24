"""One command from the raw data to the upload file.

    python run.py            # writes submissions/final.csv

Steps: build features for train and test, train the stage 1 and collective (stage 2) models with grouped
cross-validation, score the test transactions, and rank-average the stage 1 and stage 2 scores.
Seeds are fixed, so a re-run gives the same file. About 10 minutes on a 4-core laptop.
"""
import subprocess
import sys

PY = sys.executable
STEPS = [
    [PY, "-m", "src.predict", "--tag", "final_stage1", "--modules", "local", "--seeds", "3"],
    [PY, "-m", "src.stage2", "--tag", "final_stage2", "--modules", "local,episodes", "--seeds", "3", "--predict"],
    [PY, "-m", "src.blend", "--out", "final", "submissions/final_stage1.csv", "submissions/final_stage2.csv"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(">>", " ".join(step[1:]), flush=True)
        subprocess.run(step, check=True)
    print("done: submissions/final.csv")
