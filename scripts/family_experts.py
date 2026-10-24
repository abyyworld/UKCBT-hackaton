"""One expert model per laundering method, then a combiner.

    python scripts/family_experts.py --tag fam        # writes submissions/fam.csv and data/oof/fam.parquet

Each laundering method leaves a different trace (fan-in gathers from many new senders, fan-out sprays to many new
receivers, a cycle returns to its origin, bipartite and stack blocks link sets of new pairs), so one model for all
of them has to compromise. Here every method gets its own LightGBM expert: target = laundering of that method only,
trained against all non-laundering payments (laundering of other methods is left out of that expert's training, so
it does not learn to push them down). A ninth expert targets medium cases of every method (the same motifs stretched
over hours), with the slow closure features. Each expert sees a feature subset that fits its method.

The combiner is a LightGBM on the experts' out-of-fold scores, their maximum, group aggregates of the maximum
(stage 2 style) and the v17 score. All expert and combiner scores on train are out-of-fold from grouped folds
(scenario groups); family and difficulty are used only as training targets, never as features.
The script also reports how many medium cases land in the top 1% (the leaderboard's review budget).
"""
import argparse
import os
import re
import sys

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

MODULES = ["local", "episodes", "isolation", "patterns", "shape", "slow"]
EXPERTS = {
    "fan_in": ["fan_in"],
    "fan_out": ["fan_out"],
    "scatter_gather": ["fan_out_in"],
    "cycle": ["cycle"],
    "chain": ["chain", "multi_stage"],
    "block": ["bipartite", "stack"],
    "random": ["random"],
}
# feature name patterns each expert drops (directions that do not fit its shape); empty = all features
DROP = {
    "fan_in": r"(sender_out|s_out_)",
    "fan_out": r"(receiver_in|r_in_)",
}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)
TREES = 300


def fit(X, y, seed=0):
    return lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X, y), TREES)


def expert_cols(name, cols):
    pat = DROP.get(name)
    return [c for c in cols if not (pat and re.search(pat, c))]


def oof_and_test(X, y, keep, X_test, folds):
    """OOF scores on all rows (model trained on `keep` rows of the other folds) and a full-fit test score."""
    oof = np.zeros(len(y))
    for tr, ho in folds:
        tr = tr[keep[tr]]
        oof[ho] = fit(X.iloc[tr], y[tr]).predict(X.iloc[ho])
    test = fit(X[keep], y[keep]).predict(X_test)
    return oof, test


def top1_medium(score, lab):
    top = pd.Series(score, index=lab.index).rank(ascending=False) <= round(0.01 * len(lab))
    med = (lab["difficulty"] == "medium") & (lab["label"] == 1)
    return int((top & med).sum()), int(med.sum()), float(lab["label"][top].mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fam")
    ap.add_argument("--base_oof", default="data/oof/v17_realmotif.parquet")
    ap.add_argument("--base_test", default="submissions/v17_realmotif.csv")
    args = ap.parse_args()

    X, lab, groups = training_frame(MODULES)
    y = lab["label"].to_numpy()
    train_txs, _ = build_split("train", MODULES)
    test_txs, X_test = build_split("test", MODULES)
    X_test = X_test[X.columns]
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=0).split(X, y, groups))
    pos = y == 1

    E_tr, E_te = {}, {}
    for name, fams in EXPERTS.items():
        target = (pos & lab["family"].isin(fams).to_numpy()).astype(int)
        keep = ~pos | (target == 1)
        cols = expert_cols(name, X.columns)
        E_tr[name], E_te[name] = oof_and_test(X[cols], target, keep, X_test[cols], folds)
        print(f"expert {name}: {len(cols)} features, {target.sum()} positives, "
              f"OOF AP own {100 * average_precision_score(target[keep], E_tr[name][keep]):.1f}", flush=True)
    target = (pos & (lab["difficulty"] == "medium").to_numpy()).astype(int)
    keep = ~pos | (target == 1)
    E_tr["medium"], E_te["medium"] = oof_and_test(X, target, keep, X_test, folds)
    print(f"expert medium: OOF AP own {100 * average_precision_score(target[keep], E_tr['medium'][keep]):.1f}",
          flush=True)

    s_tr = pd.read_parquet(args.base_oof).set_index("tx")["oof"].reindex(lab.index).rank(pct=True).to_numpy()
    s_te = (pd.read_csv(args.base_test).set_index("TRANSACTION_HASH")["score"].reindex(test_txs["tx"])
            .rank(pct=True).to_numpy())
    A_tr = pd.DataFrame(E_tr, index=X.index).rank(pct=True)
    A_te = pd.DataFrame(E_te, index=X_test.index).rank(pct=True)
    mx_tr, mx_te = A_tr.max(axis=1).to_numpy(), A_te.max(axis=1).to_numpy()
    report(mx_tr, lab, f"{args.tag} max of experts")
    print("  top 1%%: medium %d of %d, precision %.2f" % top1_medium(mx_tr, lab))

    M_tr = pd.concat([A_tr.assign(max=mx_tr, v17=s_tr), collective_features(train_txs, mx_tr, "m_")], axis=1)
    M_te = pd.concat([A_te.assign(max=mx_te, v17=s_te), collective_features(test_txs, mx_te, "m_")], axis=1)
    M_te = M_te[M_tr.columns]
    oof = np.zeros(len(y))
    for tr, ho in folds:
        oof[ho] = fit(M_tr.iloc[tr], y[tr]).predict(M_tr.iloc[ho])
    report(oof, lab, f"{args.tag} combiner")
    print("  top 1%%: medium %d of %d, precision %.2f" % top1_medium(oof, lab))
    print("  v17 top 1%%: medium %d of %d, precision %.2f" % top1_medium(s_tr, lab))
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": oof}).to_parquet(f"data/oof/{args.tag}.parquet")
    pd.DataFrame({"tx": lab.index, **{k: v for k, v in E_tr.items()}}).to_parquet(f"data/oof/{args.tag}_experts.parquet")
    pd.DataFrame({"tx": test_txs["tx"].to_numpy(), **{k: v for k, v in E_te.items()}}).to_parquet(
        f"data/oof/{args.tag}_experts_test.parquet")
    raw = fit(M_tr, y).predict(M_te)
    write(args.tag, test_txs, raw, s_te)
