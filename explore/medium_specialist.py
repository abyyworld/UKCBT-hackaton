"""Blend a medium-only specialist model into the main model and measure overall AP.

    python -m explore.medium_specialist local,patterns,squares

Both models use the same grouped folds. The specialist is trained on medium positives against everything else
(easy and hard positives are dropped from its training rows so it learns slow schemes only).
"""
import sys

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import PARAMS, report, training_frame


def run(modules, seeds=(0, 1, 2)):
    X, lab, groups = training_frame(modules)
    y = lab["label"].to_numpy()
    med = (lab["difficulty"] == "medium").to_numpy()
    keep_spec = ~(y.astype(bool) & ~med)  # drop easy and hard positives from the specialist's training rows
    out = {}
    for seed in seeds:
        main, spec = np.zeros(len(y)), np.zeros(len(y))
        for fit, hold in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
            m = lgb.LGBMClassifier(**{**PARAMS, "random_state": seed}).fit(X.iloc[fit], y[fit])
            main[hold] = m.predict_proba(X.iloc[hold])[:, 1]
            f2 = fit[keep_spec[fit]]
            s = lgb.LGBMClassifier(**{**PARAMS, "random_state": seed, "n_estimators": 400}).fit(X.iloc[f2], med[f2])
            spec[hold] = s.predict_proba(X.iloc[hold])[:, 1]
        combos = {"main": main, "spec": spec}
        for w in (0.5, 1, 2, 4):
            combos[f"main+{w}*spec"] = main + w * spec
        combos["max"] = np.maximum(main, spec)
        rm, rs = rankdata(main) / len(y), rankdata(spec) / len(y)
        combos["rank_max"] = np.maximum(rm, rs ** 4)
        for k, v in combos.items():
            out.setdefault(k, []).append(average_precision_score(y, v))
        print(f"seed {seed}: " + "  ".join(f"{k} {100 * v[-1]:.2f}" for k, v in out.items()), flush=True)
        report(combos[max(combos, key=lambda k: out[k][-1])], lab, f"  best this seed")
    print("MEAN: " + "  ".join(f"{k} {100 * np.mean(v):.2f}" for k, v in out.items()))


if __name__ == "__main__":
    run(sys.argv[1].split(","))
