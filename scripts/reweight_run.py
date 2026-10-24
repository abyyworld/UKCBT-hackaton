"""v17 recipe trained with importance weights that make the training week look like the test week.

A LightGBM classifier learns train rows vs test rows (5-fold, out-of-fold for train rows) on features without the
amount-evenness columns and without the weekly wallet-count fingerprint (see explore/ADV_VAL.md). Each training
row gets w = p_test / (1 - p_test), clipped to [0.2, 5] and normalised to mean 1, and the v17 steps of
scripts/realmotif_run.py are run with those sample weights (stage 1, real-vs-look-alike model, stage 2).

    python scripts/reweight_run.py --tag reweight --seeds 2        # writes submissions/reweight.csv
    python scripts/reweight_run.py --tag reweight_r --robust       # also drop amount-evenness features
"""
import argparse
import os
import re
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, ".")
import src.cv as cv  # noqa: E402
from src.cv import PARAMS, oof_scores, report, training_frame  # noqa: E402
from src.features import build_split  # noqa: E402
from src.predict import fit_predict, write  # noqa: E402
from src.stage2 import MOTIF_FAMILIES, collective_features  # noqa: E402

MODULES = os.environ.get("MODULES", "local,episodes,isolation,patterns,shape").split(",")
EVEN = re.compile(r"(_usd_std$|_rel_usd$|hop_ratio_med$|hop_ratio_abs$|kept_std$|amt_z_)")
FINGERPRINT = re.compile(r"(s_out_week|s_in_week|r_in_week|r_out_week|s_distinct_dst|r_distinct_src)$")
DOMAIN_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50,
                     feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, n_estimators=300,
                     verbose=-1, n_jobs=4, random_state=0)


def domain_weights(X, X_test):
    cols = [c for c in X.columns if not EVEN.search(c) and not FINGERPRINT.search(c)]
    A = pd.concat([X[cols], X_test[cols]], ignore_index=True)
    d = np.r_[np.zeros(len(X)), np.ones(len(X_test))]
    p = np.zeros(len(A))
    for fit, hold in StratifiedKFold(5, shuffle=True, random_state=0).split(A, d):
        m = lgb.LGBMClassifier(**DOMAIN_PARAMS).fit(A.iloc[fit], d[fit])
        p[hold] = m.predict_proba(A.iloc[hold])[:, 1]
    print(f"domain classifier: {len(cols)} features, AUC {roc_auc_score(d, p):.3f}", flush=True)
    pt = p[:len(X)]
    w = np.clip(pt / np.maximum(1 - pt, 1e-6), 0.2, 5.0)
    w = w / w.mean()
    print("weights: " + " ".join(f"q{q} {np.quantile(w, q / 100):.2f}" for q in (1, 10, 50, 90, 99)), flush=True)
    return w


def subset_oof(X, y, groups, mask, seed, w):
    oof = np.zeros(len(y))
    for fit, hold in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        f = fit[mask[fit]]
        m = lgb.LGBMClassifier(**{**PARAMS, "random_state": seed}).fit(X.iloc[f], y[f], sample_weight=w[f])
        oof[hold] = m.predict_proba(X.iloc[hold])[:, 1]
    return oof


def subset_fit_predict(X, y, mask, X_test, seeds, w):
    p = np.zeros(len(X_test))
    for sd in range(seeds):
        m = lgb.LGBMClassifier(**{**PARAMS, "random_state": sd}).fit(X[mask], y[mask], sample_weight=w[mask])
        p += m.predict_proba(X_test[X.columns])[:, 1]
    return p / seeds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="reweight")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--robust", action="store_true", help="drop amount-evenness features from the models too")
    args = ap.parse_args()
    X, lab, groups = training_frame(MODULES)
    y = lab["label"].to_numpy()
    train_txs, _ = build_split("train", MODULES)
    test_txs, X_test = build_split("test", MODULES)
    X_test = X_test[X.columns]
    w = domain_weights(X, X_test)
    print("mean weight by difficulty: " + "  ".join(
        f"{k} {v:.2f}" for k, v in pd.Series(w, index=lab.index).groupby(lab["difficulty"]).mean().items()))
    if args.robust:
        keep = [c for c in X.columns if not EVEN.search(c)]
        print(f"robust: dropped {X.shape[1] - len(keep)} amount-evenness features", flush=True)
        X, X_test = X[keep], X_test[keep]
    cv.SAMPLE_WEIGHT = w  # read by cv.oof_scores and predict.fit_predict

    motif = lab["family"].isin(MOTIF_FAMILIES).to_numpy()  # family used only to choose training rows
    s1 = np.mean([oof_scores(X, y, groups, seed=sd) for sd in range(args.seeds)], axis=0)
    rm = np.mean([subset_oof(X, y, groups, motif, sd, w) for sd in range(args.seeds)], axis=0)
    report(s1, lab, f"{args.tag} stage1")
    C = pd.concat([collective_features(train_txs, s1, "susp_"), collective_features(train_txs, rm, "rm_")], axis=1)
    X2 = pd.concat([X, C], axis=1)
    oof2 = np.mean([oof_scores(X2, y, groups, seed=sd) for sd in range(args.seeds)], axis=0)
    report(oof2, lab, f"{args.tag} stage2+realmotif")
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": oof2}).to_parquet(f"data/oof/{args.tag}.parquet")

    s1_t = fit_predict(X, y, X_test, args.seeds)
    rm_t = subset_fit_predict(X, y, motif, X_test, args.seeds, w)
    C_t = pd.concat([collective_features(test_txs, s1_t, "susp_"), collective_features(test_txs, rm_t, "rm_")],
                    axis=1)
    X2_t = pd.concat([X_test, C_t], axis=1)
    raw = fit_predict(X2, y, X2_t, args.seeds)
    write(args.tag, test_txs, raw, s1_t)
