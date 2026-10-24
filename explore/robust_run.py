"""v17 without the amount-evenness features, to test whether they hurt on test.

    python -m explore.robust_run --tag v17r --seeds 2      # writes submissions/v17r.csv

In train, laundering groups have even amounts (coefficient of variation 0.45 vs 0.91 for normal fast groups), but
in test the amount spread inside fast groups is clearly higher (lower quartile 0.64 vs 0.45), which suggests test
laundering with deliberately uneven amounts. This runs scripts/realmotif_run.py unchanged, except that features
describing how even amounts are (spread of amounts in an episode, amount relative to the episode, hop-to-hop amount
ratios, spread of kept fractions, amount z-scores) are removed. Tempo and value balance features stay.
"""
import re
import runpy
import sys

import src.cv as cv
import src.features as features

DROP = re.compile(r"(_usd_std$|_rel_usd$|hop_ratio_med$|hop_ratio_abs$|kept_std$|amt_z_)")

_build_split = features.build_split
_training_frame = cv.training_frame


def build_split(split, modules=None, refresh=False):
    txs, X = _build_split(split, modules, refresh)
    return txs, X.loc[:, [c for c in X.columns if not DROP.search(c)]]


def training_frame(modules=None, refresh=None):
    X, lab, groups = _training_frame(modules, refresh)
    keep = [c for c in X.columns if not DROP.search(c)]
    print(f"robust_run: dropped {X.shape[1] - len(keep)} amount-evenness features, kept {len(keep)}", flush=True)
    return X.loc[:, keep], lab, groups


features.build_split = build_split
cv.training_frame = training_frame

if __name__ == "__main__":
    sys.argv = ["scripts/realmotif_run.py"] + sys.argv[1:]
    runpy.run_path("scripts/realmotif_run.py", run_name="__main__")
