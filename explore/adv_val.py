"""Adversarial validation: can a model tell train rows from test rows using the local features?

    python -m explore.adv_val

High AUC for a feature means its distribution shifts between the splits, so the model may not transfer.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.load import load_split

from src.features import build_split

SEED = 0


def main():
    _, Xtr = build_split("train", ["local"])
    _, Xte = build_split("test", ["local"])
    X = pd.concat([Xtr, Xte], ignore_index=True)
    y = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]

    # Wallets never cross splits, so per-wallet counts act as a fingerprint. Grouping folds by sender stops the
    # model from recognising wallets it has already seen, which leaves only genuine distribution shift.
    senders = np.r_[load_split("train")[0]["sender"].to_numpy(), load_split("test")[0]["sender"].to_numpy()]
    for name, folds, groups in [("random folds", StratifiedKFold(5, shuffle=True, random_state=SEED), None),
                                ("folds grouped by sender", StratifiedGroupKFold(5, shuffle=True, random_state=SEED),
                                 senders)]:
        oof, imp = np.zeros(len(y)), np.zeros(X.shape[1])
        for fit, hold in folds.split(X, y, groups):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED)
            m.fit(X.iloc[fit], y[fit])
            oof[hold] = m.predict_proba(X.iloc[hold])[:, 1]
            imp += m.booster_.feature_importance("gain")
        print(f"train vs test AUC, {name}: {roc_auc_score(y, oof):.3f}")

    per = {}
    for c in X.columns:
        v = X[c].fillna(-1e9).to_numpy()
        a = roc_auc_score(y, v)
        per[c] = max(a, 1 - a)
    res = pd.DataFrame({"gain": imp / imp.sum(), "auc_alone": pd.Series(per)}).sort_values("gain", ascending=False)
    print(res.head(15).round(3).to_string())
    print("\nfeatures with single-feature AUC > 0.6:")
    print(res[res.auc_alone > 0.6].sort_values("auc_alone", ascending=False).round(3).to_string())
    return res


if __name__ == "__main__":
    main()
