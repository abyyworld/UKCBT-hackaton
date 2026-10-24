"""A different learner: scikit-learn HistGradientBoosting on the stage 1 features.

Reports out-of-fold AP alone and in a rank blend with the v17 out-of-fold scores, then fits on all of train and
writes submissions/hgb.csv.
    python scripts/hgb_run.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
from src.cv import report, training_frame  # noqa: E402
from src.features import build_split  # noqa: E402
from src.predict import write  # noqa: E402

MODULES = "local,episodes,isolation,patterns,shape".split(",")


def model(seed):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20,
                                          l2_regularization=1.0, max_features=0.7, early_stopping=False,
                                          random_state=seed)


if __name__ == "__main__":
    X, lab, groups = training_frame(MODULES)
    y = lab["label"].to_numpy()
    oof = np.zeros(len(y))
    for fit, hold in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
        oof[hold] = model(0).fit(X.iloc[fit], y[fit]).predict_proba(X.iloc[hold])[:, 1]
    report(oof, lab, "hgb alone")
    pd.DataFrame({"tx": lab.index, "oof": oof}).to_parquet("data/oof/hgb.parquet")
    v17 = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "data/oof/final.parquet").set_index("tx").loc[lab.index, "oof"]
    r17, rh = pd.Series(v17.to_numpy()).rank(pct=True), pd.Series(oof).rank(pct=True)
    report(r17.to_numpy(), lab, "v17 alone")
    for w in (0.25, 0.5):
        report(((1 - w) * r17 + w * rh).to_numpy(), lab, f"rank blend v17 + {w} hgb")
    test_txs, Xt = build_split("test", MODULES)
    m = model(0).fit(X, y)
    raw = m.predict_proba(Xt[X.columns])[:, 1]
    write("hgb", test_txs, raw, raw)
    print("AP check", round(100 * average_precision_score(y, oof), 2))
