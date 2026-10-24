"""Stage 2 with an extra real-vs-look-alike motif model (idea from the second session).

The extra model is trained only on rows from motif families (family used only to choose training rows), so it
learns what separates planted motifs from their normal look-alikes. Its scores feed stage 2 as collective features.
    python scripts/realmotif_run.py --tag v17 --seeds 2
"""
import argparse
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
import src.cv as cv  # noqa: E402
from src.cv import PARAMS, oof_scores, report, training_frame  # noqa: E402
from src.features import build_split  # noqa: E402
from src.predict import fit_predict, write  # noqa: E402
from src.stage2 import MOTIF_FAMILIES, RELAY_FAMILIES, collective_features  # noqa: E402

MODULES = ["local", "episodes", "isolation", "patterns", "shape"]


def subset_oof(X, y, groups, mask, seed):
    oof = np.zeros(len(y))
    for fit, hold in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        f = fit[mask[fit]]
        m = lgb.LGBMClassifier(**{**PARAMS, "random_state": seed}).fit(X.iloc[f], y[f])
        oof[hold] = m.predict_proba(X.iloc[hold])[:, 1]
    return oof


def subset_fit_predict(X, y, mask, X_test, seeds):
    p = np.zeros(len(X_test))
    for sd in range(seeds):
        m = lgb.LGBMClassifier(**{**PARAMS, "random_state": sd}).fit(X[mask], y[mask])
        p += m.predict_proba(X_test[X.columns])[:, 1]
    return p / seeds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--relay", action="store_true", help="also add a real-vs-normal model trained on relay-hub rows")
    args = ap.parse_args()
    X, lab, groups = training_frame(MODULES)
    y = lab["label"].to_numpy()
    motif = lab["family"].isin(MOTIF_FAMILIES).to_numpy()
    train_txs, _ = build_split("train", MODULES)
    s1 = np.mean([oof_scores(X, y, groups, seed=sd) for sd in range(args.seeds)], axis=0)
    rm = np.mean([subset_oof(X, y, groups, motif, sd) for sd in range(args.seeds)], axis=0)
    report(s1, lab, f"{args.tag} stage1")
    parts = [collective_features(train_txs, s1, "susp_"), collective_features(train_txs, rm, "rm_")]
    relay = lab["family"].isin(RELAY_FAMILIES).to_numpy()
    if args.relay:
        rr = np.mean([subset_oof(X, y, groups, relay, sd) for sd in range(args.seeds)], axis=0)
        parts.append(collective_features(train_txs, rr, "rr_"))
    C = pd.concat(parts, axis=1)
    X2 = pd.concat([X, C], axis=1)
    oof2 = np.mean([oof_scores(X2, y, groups, seed=sd) for sd in range(args.seeds)], axis=0)
    report(oof2, lab, f"{args.tag} stage2+realmotif")
    pd.DataFrame({"tx": lab.index, "oof": oof2}).to_parquet(f"data/oof/{args.tag}.parquet")
    test_txs, X_test = build_split("test", MODULES)
    s1_t = fit_predict(X, y, X_test, args.seeds)
    rm_t = subset_fit_predict(X, y, motif, X_test, args.seeds)
    parts_t = [collective_features(test_txs, s1_t, "susp_"), collective_features(test_txs, rm_t, "rm_")]
    if args.relay:
        parts_t.append(collective_features(test_txs, subset_fit_predict(X, y, relay, X_test, args.seeds), "rr_"))
    C_t = pd.concat(parts_t, axis=1)
    X2_t = pd.concat([X_test, C_t], axis=1)
    raw = fit_predict(X2, y, X2_t, args.seeds)
    write(args.tag, test_txs, raw, s1_t)
