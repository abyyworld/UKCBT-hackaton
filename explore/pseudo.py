"""Transductive pseudo-labelling check, measured inside grouped CV.

For each fold the held-out fold plays the unlabelled test set: fit on the training folds, score held-out,
add its top k as pseudo-positives and bottom m as pseudo-negatives (weight w), refit, rescore held-out.
Held-out labels are used only for the final AP.

    python -m explore.pseudo --modules local --seeds 0,1,2
"""
import argparse
import itertools
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import PARAMS, training_frame


def pseudo_fit_predict(X, y, X_unl, k=100, m=5000, weight=1.0, rounds=1, seed=0, params=None, plain=None):
    """Fit on (X, y), then add the k most and m least suspicious rows of X_unl as pseudo-labels and refit.
    Returns the plain scores and the pseudo scores for X_unl."""
    params = {**PARAMS, **(params or {}), "random_state": seed}
    if plain is None:
        plain = lgb.LGBMClassifier(**params).fit(X, y).predict_proba(X_unl[X.columns])[:, 1]
    score = plain
    for _ in range(rounds):
        order = np.argsort(-score)
        pos, neg = order[:k], order[len(order) - m:] if m else order[:0]
        Xp = pd.concat([X, X_unl.iloc[pos][X.columns], X_unl.iloc[neg][X.columns]])
        yp = np.r_[y, np.ones(len(pos)), np.zeros(len(neg))]
        wp = np.r_[np.ones(len(y)), np.full(len(pos) + len(neg), weight)]
        model = lgb.LGBMClassifier(**params).fit(Xp, yp, sample_weight=wp)
        score = model.predict_proba(X_unl[X.columns])[:, 1]
    return plain, score


def pseudo_predict(X, y, X_test, seeds=3, frac=0.0125, rounds=2, weight=1.0, params=None):
    """Drop-in for src.predict.fit_predict: per seed, fit on train, then `rounds` times add the top `frac` of the
    test rows as weight-`weight` pseudo-positives and refit. frac 0.0125 matches k=100 in a CV fold (about 500
    of the 40,101 test rows). No pseudo-negatives."""
    k = int(round(frac * len(X_test)))
    pred = np.zeros(len(X_test))
    for seed in range(seeds):
        pred += pseudo_fit_predict(X, y, X_test, k, 0, weight, rounds, seed, params)[1]
    return pred / seeds


def ap_rows(y, s, lab):
    out = {"ALL": average_precision_score(y, s)}
    for diff in ("easy", "medium", "hard"):
        msk = lab["difficulty"].isin([diff, "normal"]).to_numpy()
        out[diff] = average_precision_score(y[msk], s[msk])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="local")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ks", default="100,300")
    ap.add_argument("--ms", default="0,5000")
    ap.add_argument("--weights", default="0.5,1")
    ap.add_argument("--rounds", default="1")
    args = ap.parse_args()
    X, lab, groups = training_frame(args.modules.split(","))
    y = lab["label"].to_numpy()
    print(f"rows {len(y)}  positives {y.sum()}  features {X.shape[1]}", flush=True)
    configs = list(itertools.product([int(v) for v in args.ks.split(",")], [int(v) for v in args.ms.split(",")],
                                     [float(v) for v in args.weights.split(",")], [int(v) for v in args.rounds.split(",")]))
    res = []
    for seed in [int(v) for v in args.seeds.split(",")]:
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups))
        plain = np.zeros(len(y))
        pseudo = {c: np.zeros(len(y)) for c in configs}
        for fit, hold in folds:
            t0 = time.time()
            p = None
            for c in configs:
                k, m, w, r = c
                p, s = pseudo_fit_predict(X.iloc[fit], y[fit], X.iloc[hold], k, m, w, r, seed, plain=p)
                plain[hold], pseudo[c][hold] = p, s
            print(f"  seed {seed} fold done in {time.time() - t0:.0f}s", flush=True)
        base = ap_rows(y, plain, lab)
        res.append(dict(seed=seed, k=0, m=0, w=0, rounds=0, **base))
        print(f"seed {seed} plain " + "  ".join(f"{a} {100 * b:.2f}" for a, b in base.items()), flush=True)
        for c, s in pseudo.items():
            r = ap_rows(y, s, lab)
            res.append(dict(seed=seed, k=c[0], m=c[1], w=c[2], rounds=c[3], **r))
            print(f"seed {seed} k={c[0]} m={c[1]} w={c[2]} r={c[3]} "
                  + "  ".join(f"{a} {100 * b:.2f} ({100 * (b - base[a]):+.2f})" for a, b in r.items()), flush=True)
    df = pd.DataFrame(res)
    df[["ALL", "easy", "medium", "hard"]] *= 100
    summ = df.groupby(["k", "m", "w", "rounds"])[["ALL", "easy", "medium", "hard"]].mean()
    summ["gain_ALL"] = summ["ALL"] - summ.loc[(0, 0, 0, 0), "ALL"]
    print(summ.round(2).to_string())
    df.to_csv(f"data/pseudo_{args.modules.replace(',', '_')}.csv", index=False)
