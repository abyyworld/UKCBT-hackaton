"""Model settings and blends on the same features and folds, three seeds each.

    python -m explore.models [modules]

Prints mean out-of-fold AP for LightGBM variants, an ExtraTrees model and rank-average blends.
"""
import sys

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import PARAMS, training_frame

VARIANTS = {
    "lgb": {},
    "lgb_small": dict(num_leaves=15, min_child_samples=40, n_estimators=900),
    "lgb_slow": dict(learning_rate=0.015, n_estimators=1400),
    "lgb_balanced": dict(scale_pos_weight=5.0),
    "lgb_ff": dict(feature_fraction=0.4, lambda_l2=5.0),
}


def oof(make, X, y, groups, seed):
    out = np.zeros(len(y))
    for fit, hold in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        m = make(seed)
        m.fit(X.iloc[fit], y[fit])
        out[hold] = m.predict_proba(X.iloc[hold])[:, 1]
    return out


def main(modules):
    X, lab, groups = training_frame(modules)
    y = lab["label"].to_numpy()
    Xf = X.fillna(-999)
    makers = {k: (lambda v: lambda s: lgb.LGBMClassifier(**{**PARAMS, **v, "random_state": s}))(v)
              for k, v in VARIANTS.items()}
    makers["et"] = lambda s: ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.3, n_jobs=4, random_state=s)
    res = {k: [] for k in list(makers) + ["blend_lgb_et", "blend_all_lgb", "blend_all"]}
    for seed in (0, 1, 2):
        sc = {k: oof(f, Xf if k == "et" else X, y, groups, seed) for k, f in makers.items()}
        for k, v in sc.items():
            res[k].append(average_precision_score(y, v))
        r = {k: rankdata(v) for k, v in sc.items()}
        res["blend_lgb_et"].append(average_precision_score(y, r["lgb"] + r["et"]))
        res["blend_all_lgb"].append(average_precision_score(y, sum(r[k] for k in VARIANTS)))
        res["blend_all"].append(average_precision_score(y, sum(r.values())))
        print(f"seed {seed}: " + "  ".join(f"{k} {100 * v[-1]:.2f}" for k, v in res.items()), flush=True)
    print("MEAN: " + "  ".join(f"{k} {100 * np.mean(v):.2f}" for k, v in res.items()))


if __name__ == "__main__":
    main((sys.argv[1] if len(sys.argv) > 1 else "local").split(","))
