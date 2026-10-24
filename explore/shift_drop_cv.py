"""Out-of-fold AP when dropping feature groups flagged by the shift audit (one seed).

    python -m explore.shift_drop_cv <name> <regex or ->
"""
import re
import sys

from src import cv

MODS = "local,episodes,isolation,patterns,shape,slow".split(",")
GROUPS = {
    "giant": r"^episodes__novel(7200|21600)_",
    "amount": r"(usd_std|usd_range|drift_abs|kept_std|kept_absmax|hop_ratio_abs)$",
}

if __name__ == "__main__":
    name, pat = sys.argv[1], sys.argv[2]
    X, lab, groups = cv.training_frame(MODS)
    if pat != "-":
        pat = "|".join(f"(?:{GROUPS.get(p, p)})" for p in pat.split("+"))
        drop = [c for c in X.columns if re.search(pat, c)]
        X = X.drop(columns=drop)
        print(f"dropped {len(drop)}")
    print(f"features: {X.shape[1]}")
    oof = cv.oof_scores(X, lab["label"].to_numpy(), groups)
    cv.report(oof, lab, name)
