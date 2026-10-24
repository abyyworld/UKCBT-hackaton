"""A different learner: CatBoost (ordered boosting, symmetric trees) on the v17 inputs plus group scores from v17.

Same inputs and stacking protocol as scripts/extratrees_run.py, so it can be scored alone or rank-blended.
    python scripts/catboost_run.py --tag cb        # writes submissions/cb.csv and data/oof/cb.parquet
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
from src.cv import report, training_frame  # noqa: E402
from src.features import build_split  # noqa: E402
from src.predict import write  # noqa: E402
from src.stage2 import collective_features  # noqa: E402

MODULES = os.environ.get("MODULES", "local,episodes,isolation,patterns,shape").split(",")


def model(seed, iters):
    return CatBoostClassifier(iterations=iters, depth=6, learning_rate=0.05, l2_leaf_reg=3, thread_count=4,
                              random_seed=seed, verbose=0, allow_writing_files=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cb")
    ap.add_argument("--base_oof", default="data/oof/v17_realmotif.parquet")
    ap.add_argument("--base_test", default="submissions/v17_realmotif.csv")
    ap.add_argument("--iters", type=int, default=800)
    args = ap.parse_args()
    X, lab, groups = training_frame(MODULES)
    y = lab["label"].to_numpy()
    train_txs, _ = build_split("train", MODULES)
    test_txs, X_test = build_split("test", MODULES)
    s_tr = pd.read_parquet(args.base_oof).set_index("tx")["oof"].reindex(lab.index).to_numpy()
    s_te = pd.read_csv(args.base_test).set_index("TRANSACTION_HASH")["score"].reindex(test_txs["tx"]).to_numpy()
    Xa = pd.concat([X, collective_features(train_txs, s_tr, "b_")], axis=1)
    Xb = pd.concat([X_test, collective_features(test_txs, s_te, "b_")], axis=1)[Xa.columns]
    oof = np.zeros(len(y))
    for fit, hold in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(Xa, y, groups):
        oof[hold] = model(0, args.iters).fit(Xa.iloc[fit], y[fit]).predict_proba(Xa.iloc[hold])[:, 1]
    report(oof, lab, f"{args.tag} catboost (3-fold)")
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": oof}).to_parquet(f"data/oof/{args.tag}.parquet")
    raw = model(0, args.iters).fit(Xa, y).predict_proba(Xb)[:, 1]
    write(args.tag, test_txs, raw, s_te)
