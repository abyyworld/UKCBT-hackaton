"""Two-stage cascade re-ranker, the way search engines rank results.

Stage one is v17: it already puts almost every suspicious payment in its top few thousand, and its mistakes sit
inside that top region (fast normal look-alikes above medium laundering, relay noise). Stage two is a LightGBM
trained ONLY on the rows v17 ranks in its top K, so all of its capacity goes to telling real laundering from
look-alikes among hard-to-separate candidates. Its inputs are the feature modules (including the slow closure
features that help medium cases), the v17 score itself and group/neighbour aggregates of the v17 score.
On train the v17 scores are out-of-fold, on test they are the v17 test scores.

Final ranking: rows outside the top K keep their v17 order, below everything inside the top K. Rows inside the
top K are ordered by alpha * rank(cascade) + (1 - alpha) * rank(v17). The cascade out-of-fold scores come from
3 grouped folds over the top-K train rows only. K and alpha are picked by out-of-fold AP over the whole list.
    python scripts/cascade_run.py --tag cascade
"""
import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
from src.cv import report, training_frame  # noqa: E402
from src.features import build_split  # noqa: E402
from src.predict import write  # noqa: E402
from src.stage2 import collective_features  # noqa: E402

MODULES = os.environ.get("MODULES", "local,episodes,isolation,patterns,shape,slow").split(",")
PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=1)
PARAMS.update(json.loads(os.environ.get("CASCADE_PARAMS", "{}")))


def fit_model(X, y, seed=0):
    return lgb.LGBMClassifier(**{**PARAMS, "random_state": seed}).fit(X, y)


def rerank(base, top_idx, cascade_top, alpha):
    """Whole-list score: top-K rows above all others, ordered by the alpha mix of ranks; the rest keep v17 order."""
    n = len(base)
    out = rankdata(base) / n  # in (0, 1]
    mix = alpha * rankdata(cascade_top) + (1 - alpha) * rankdata(base[top_idx])
    out[top_idx] = 2.0 + rankdata(mix + 1e-9 * rankdata(base[top_idx])) / len(top_idx)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cascade")
    ap.add_argument("--base_oof", default="data/oof/v17_realmotif.parquet")
    ap.add_argument("--base_test", default="submissions/v17_realmotif.csv")
    ap.add_argument("--ks", default="2000,3000,4000")
    ap.add_argument("--alphas", default="0.5,0.75,1.0")
    ap.add_argument("--dry", action="store_true", help="evaluate only, write nothing")
    args = ap.parse_args()
    X, lab, groups = training_frame(MODULES)
    y = lab["label"].to_numpy()
    train_txs, _ = build_split("train", MODULES)
    test_txs, X_test = build_split("test", MODULES)
    s_tr = pd.read_parquet(args.base_oof).set_index("tx")["oof"].reindex(lab.index).to_numpy()
    s_te = pd.read_csv(args.base_test).set_index("TRANSACTION_HASH")["score"].reindex(test_txs["tx"]).to_numpy()
    assert np.isfinite(s_tr).all() and np.isfinite(s_te).all(), "missing base scores"
    # train OOF scores are probabilities, the test file holds rank-normalised scores: put both on the percentile
    # scale so the score feature and its group aggregates mean the same thing on train and test
    p_tr, p_te = rankdata(s_tr) / len(s_tr), rankdata(s_te) / len(s_te)
    Xa = pd.concat([X, collective_features(train_txs, p_tr, "b_")], axis=1)
    Xa["v17_score"] = p_tr
    Xb = pd.concat([X_test, collective_features(test_txs, p_te, "b_")], axis=1)
    Xb["v17_score"] = p_te
    if os.environ.get("CASCADE_NO_COLLECTIVE"):
        Xa = Xa.drop(columns=[c for c in Xa.columns if c.startswith("b_")])
    Xb = Xb[Xa.columns]
    print(f"features: {Xa.shape[1]}, train {len(Xa)}, test {len(Xb)}")
    report(s_tr, lab, "v17 base")

    order_tr = np.argsort(-s_tr, kind="stable")
    results = {}
    cascade_oof = {}
    for K in [int(k) for k in args.ks.split(",")]:
        top = order_tr[:K]
        yk, gk = y[top], groups[top]
        print(f"K={K}: positives in top K {yk.sum()} of {y.sum()}")
        oof_k = np.zeros(K)
        for fit, hold in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(Xa.iloc[top], yk, gk):
            oof_k[hold] = fit_model(Xa.iloc[top[fit]], yk[fit]).predict_proba(Xa.iloc[top[hold]])[:, 1]
        cascade_oof[K] = oof_k
        print(f"  within-top-K AP x100: v17 {100 * average_precision_score(yk, s_tr[top]):.2f}"
              f"  cascade {100 * average_precision_score(yk, oof_k):.2f}")
        for alpha in [float(a) for a in args.alphas.split(",")]:
            full = rerank(s_tr, top, oof_k, alpha)
            results[(K, alpha)] = report(full, lab, f"K={K} alpha={alpha}")

    (K, alpha), best = max(results.items(), key=lambda kv: kv[1])
    print(f"chosen K={K} alpha={alpha}: OOF ALL {100 * best:.2f}")
    top = order_tr[:K]
    final_oof = rerank(s_tr, top, cascade_oof[K], alpha)
    report(final_oof, lab, f"{args.tag} chosen")
    blend = (rankdata(final_oof) + rankdata(s_tr)) / 2
    report(blend, lab, f"{args.tag} 50/50 rank blend with v17")
    if args.dry:
        sys.exit(0)
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": final_oof}).to_parquet(f"data/oof/{args.tag}.parquet")

    model = fit_model(Xa.iloc[top], y[top])
    top_te = np.argsort(-s_te, kind="stable")[:K]
    raw = rerank(s_te, top_te, model.predict_proba(Xb.iloc[top_te])[:, 1], alpha)
    write(args.tag, test_txs, raw, s_te)
