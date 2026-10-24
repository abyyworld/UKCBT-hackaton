"""One command from the raw data to the upload file.

    python run.py            # writes submissions/final.csv

Steps: build features for train and test, train the stage 1 and collective (stage 2) models with grouped
cross-validation, then score the test transactions with the stage 2 model.
Seeds are fixed, so a re-run gives the same file. About 10 minutes on a 4-core laptop.
"""
import subprocess
import sys

PY = sys.executable
MODULES = "local,episodes,isolation,patterns,shape"
STEPS = [
    [PY, "-m", "src.stage2", "--tag", "final", "--modules", MODULES, "--seeds", "2", "--predict"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(">>", " ".join(step[1:]), flush=True)
        subprocess.run(step, check=True)
    print("done: submissions/final.csv")
