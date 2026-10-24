"""Learning to rank: LightGBM with a ranking objective instead of log loss.

The leaderboard metric is Average Precision, which only cares about the order of the scores. Our usual models
minimise log loss, which also spends effort on calibrating probabilities. Here LightGBM is trained with a ranking
objective (lambdarank and/or rank_xendcg) so the trees are fitted directly to put suspicious payments above normal ones.

Inputs are exactly those of scripts/extratrees_run.py: the v17 feature modules plus collective features built from
v17's scores (out-of-fold v17 scores on train, v17 test scores on test).

Queries: rows are sorted by timestamp and cut into contiguous time blocks (default 2 hours; 1h 60.84, 2h 62.01, 6h 61.81, 24h 61.01 OOF ALL). Each block is one query.
Group sizes are passed in that sorted order and predictions are put back into the original row order afterwards.
lambdarank gets a high truncation level because many positives share a block. Ranking scores are not probabilities,
so they are min-max scaled to [0, 1] before writing.

For a fair comparison a plain binary LightGBM on the same inputs is run in the same 3 grouped folds. Every objective
is reported, together with a 50/50 rank average against v17's OOF, and the objective with the best OOF ALL is fitted
on all of train and written as submissions/<tag>.csv.
    python scripts/ltr_run.py --tag ltr --objs lambdarank,binary
"""
import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
from src.cv import report, training_frame  # noqa: E402
from src.features import build_split  # noqa: E402
from src.predict import write  # noqa: E402
from src.stage2 import collective_features  # noqa: E402

MODULES = os.environ.get("MODULES", "local,episodes,isolation,patterns,shape").split(",")

BASE = dict(learning_rate=0.05, n_estimators=400, num_leaves=31, min_child_samples=20, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=1)


def params(obj, seed):
    p = {**BASE, "objective": obj, "random_state": seed}
    if obj == "lambdarank":
        p.update(lambdarank_truncation_level=int(os.environ.get("TRUNC", 300)), label_gain=[0, 1])
    if obj in ("lambdarank", "rank_xendcg"):
        p["eval_at"] = [100]
    return p


def query_order(ts, hours):
    """Row order sorted by time, and the group sizes of contiguous time blocks in that order."""
    order = np.argsort(ts, kind="stable")
    block = (ts[order] - ts.min()) // int(hours * 3600)
    _, sizes = np.unique(block, return_counts=True)  # blocks are already sorted, so counts follow the order
    return order, sizes


def fit_predict(obj, X, y, ts, X_new, hours, seed=0):
    p = params(obj, seed)
    if obj == "binary":
        m = lgb.LGBMClassifier(**p).fit(X, y)
        return m.predict_proba(X_new)[:, 1]
    order, sizes = query_order(ts, hours)
    m = lgb.LGBMRanker(**p).fit(X.iloc[order], y[order], group=sizes)
    return m.predict(X_new)  # prediction is row-wise, so X_new keeps its own order


def minmax(s):
    s = np.asarray(s, dtype=float)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def rank_avg(a, b):
    return (pd.Series(a).rank().to_numpy() + pd.Series(b).rank().to_numpy()) / 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ltr")
    ap.add_argument("--objs", default="lambdarank,binary")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--base_oof", default="data/oof/v17_realmotif.parquet")
    ap.add_argument("--base_test", default="submissions/v17_realmotif.csv")
    args = ap.parse_args()
    t0 = time.time()
    X, lab, groups = training_frame(MODULES)
    y = lab["label"].to_numpy()
    train_txs, _ = build_split("train", MODULES)
    test_txs, X_test = build_split("test", MODULES)
    s_tr = pd.read_parquet(args.base_oof).set_index("tx")["oof"].reindex(lab.index).to_numpy()
    s_te = pd.read_csv(args.base_test).set_index("TRANSACTION_HASH")["score"].reindex(test_txs["tx"]).to_numpy()
    Xa = pd.concat([X, collective_features(train_txs, s_tr, "b_")], axis=1).fillna(-1)
    Xb = pd.concat([X_test, collective_features(test_txs, s_te, "b_")], axis=1)[Xa.columns].fillna(-1)
    ts_tr = train_txs.set_index("tx")["ts"].reindex(lab.index).to_numpy().astype(np.int64)
    ts_te = test_txs["ts"].to_numpy().astype(np.int64)
    print(f"inputs: {Xa.shape[1]} features, {len(y)} train rows, {int(y.sum())} positives ({time.time() - t0:.0f}s)")

    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=0).split(Xa, y, groups))
    results = {}
    for obj in args.objs.split(","):
        oof = np.zeros(len(y))
        for fit, hold in folds:
            raw = fit_predict(obj, Xa.iloc[fit], y[fit], ts_tr[fit], Xa.iloc[hold], args.hours)
            oof[hold] = minmax(raw)  # each fold scaled to [0, 1] so folds are comparable
        ap_all = report(oof, lab, f"{args.tag} {obj} (3-fold, {args.hours:g}h queries)")
        blend = average_precision_score(y, rank_avg(oof, s_tr))
        print(f"  {obj}: 50/50 rank average with v17 OOF AP x100: {100 * blend:.2f}  ({time.time() - t0:.0f}s)")
        results[obj] = (ap_all, oof)

    best = max(results, key=lambda k: results[k][0])
    oof = results[best][1]
    print(f"best objective by OOF ALL: {best}")
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": oof}).to_parquet(f"data/oof/{args.tag}.parquet")
    raw = minmax(fit_predict(best, Xa, y, ts_tr, Xb, args.hours))
    write(args.tag, test_txs, raw, s_te)
    print(f"done in {time.time() - t0:.0f}s")
