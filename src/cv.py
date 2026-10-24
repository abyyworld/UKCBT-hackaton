"""Out-of-fold evaluation, grouped by scenario so a planted motif never straddles folds.

    python -m src.cv [--modules local,paths] [--refresh name] [--tag v1]
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.features import MODULES, build_split
from src.load import load_labels

# Optional training weights by difficulty (used only to weight training rows, never as a feature).
# AP counts every suspicious case equally, but the loss is dominated by the 800 easy ones.
WEIGHTS = {k: float(os.environ.get(f"W_{k.upper()}", 1)) for k in ("easy", "medium", "hard")}
SAMPLE_WEIGHT = None  # set by training_frame, aligned with the training rows

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=700, verbose=-1, n_jobs=4)


def training_frame(modules=None, refresh=None):
    for name in refresh or []:
        build_split("train", [name], refresh=True)
    txs, X = build_split("train", modules)
    lab = load_labels().set_index("tx").loc[txs["tx"]]
    global SAMPLE_WEIGHT
    SAMPLE_WEIGHT = lab["difficulty"].map(WEIGHTS).fillna(1.0).to_numpy()
    groups = lab["scenario_id"].fillna(pd.Series(lab.index, index=lab.index)).to_numpy()
    return X, lab, groups


def oof_scores(X, y, groups, seed=0, n_splits=5, params=None):
    params = {**PARAMS, **(params or {}), "random_state": seed}
    oof = np.zeros(len(y))
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit, hold in folds.split(X, y, groups):
        model = lgb.LGBMClassifier(**params)
        sw = SAMPLE_WEIGHT[fit] if SAMPLE_WEIGHT is not None and len(SAMPLE_WEIGHT) == len(y) else None
        model.fit(X.iloc[fit], y[fit], sample_weight=sw)
        oof[hold] = model.predict_proba(X.iloc[hold])[:, 1]
    return oof


def report(oof, lab, title=""):
    y = lab["label"].to_numpy()
    rows = {"ALL": average_precision_score(y, oof)}
    for diff in ("easy", "medium", "hard"):
        m = lab["difficulty"].isin([diff, "normal"]).to_numpy()
        rows[diff] = average_precision_score(y[m], oof[m])
    print(f"{title} AP x100: " + "  ".join(f"{k} {100 * v:.2f}" for k, v in rows.items()))
    fam = {}
    for name, g in lab.assign(s=oof).groupby("family"):
        if g["label"].nunique() == 2:
            fam[name] = average_precision_score(g["label"], g["s"])
    print("  per-family AP (within family): " + "  ".join(f"{k} {100 * v:.0f}" for k, v in sorted(fam.items())))
    return rows["ALL"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--refresh", default="")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()
    modules = args.modules.split(",")
    X, lab, groups = training_frame(modules, [m for m in args.refresh.split(",") if m])
    print(f"features: {X.shape[1]}")
    oof = oof_scores(X, lab["label"].to_numpy(), groups)
    report(oof, lab, args.tag)
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": oof}).to_parquet(f"data/oof/{args.tag}.parquet")
