"""v17 recipe with every feature the shift audit flagged removed (explore/SHIFT.md, explore/NOTES.md).

Two train-to-test shifts are removed:
1. Amount evenness: in the test week laundering uses uneven amounts, so features rewarding even amounts mislead.
2. Giant components: first-time-pair episodes at 2 h and 6 h merge into one blob covering about a third of the
   week, and rows inside get week-specific constants (a train vs test fingerprint). Those scales are dropped from
   the features and from stage 2 group scoring.
    python scripts/robust2_run.py --tag v27 --seeds 2
"""
import re
import runpy
import sys

sys.path.insert(0, ".")
import src.cv as cv  # noqa: E402
import src.features as features  # noqa: E402
import src.stage2 as stage2  # noqa: E402

DROP = re.compile(r"(^episodes__novel(7200|21600)_|(usd_std|usd_range|drift_abs|kept_std|kept_absmax|hop_ratio_abs)$"
                  r"|_rel_usd$|hop_ratio_med$|amt_z_)")
stage2.SCALES = [sc for sc in stage2.SCALES if not (sc[0] == "novel" and sc[1] >= 7200)]

_build_split, _training_frame = features.build_split, cv.training_frame


def build_split(split, modules=None, refresh=False):
    txs, X = _build_split(split, modules, refresh)
    return txs, X.loc[:, [c for c in X.columns if not DROP.search(c)]]


def training_frame(modules=None, refresh=None):
    X, lab, groups = _training_frame(modules, refresh)
    keep = [c for c in X.columns if not DROP.search(c)]
    print(f"robust2: dropped {X.shape[1] - len(keep)} shifted features, kept {len(keep)}", flush=True)
    return X.loc[:, keep], lab, groups


features.build_split, cv.training_frame = build_split, training_frame

if __name__ == "__main__":
    sys.argv = ["scripts/realmotif_run.py"] + sys.argv[1:]
    runpy.run_path("scripts/realmotif_run.py", run_name="__main__")
