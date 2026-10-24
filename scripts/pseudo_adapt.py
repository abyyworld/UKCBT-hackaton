"""Test-week adaptation by pseudo-labelling, cross-fitted on test.

    python scripts/pseudo_adapt.py [--base base] [--top 500] [--bottom 0.8] [--weight 0.5] [--seeds 2]

1. Current test scores come from submissions/<base>.csv. If the file is missing, it is made with
   scripts/realmotif_run.py --tag <base> --seeds 2 (about 10 minutes).
2. Pseudo-labels on test: the top --top rows by score are 1, the bottom --bottom share are 0, the middle is unused.
3. Cross-fit: test is split into 5 folds grouped by connected components of the all300 episodes, so a scheme never
   straddles folds. For each fold, LightGBM is trained on all train rows (true labels, weight 1) plus the
   pseudo-labelled test rows of the other 4 folds (weight --weight), and predicts the held-out fold. No test row is
   scored by a model that saw its own pseudo-label.
4. Features: modules local, episodes, isolation, patterns, shape, minus the amount-evenness columns (amounts are
   much more uneven in test than in train, see explore/robust_run.py).
Writes submissions/pseudo.csv.
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.cv import PARAMS
from src.features import build_split
from src.features.episodes import _components
from src.load import load_labels
from src.predict import write

MODULES = ["local", "episodes", "isolation", "patterns", "shape"]
DROP = re.compile(r"(_usd_std$|_rel_usd$|hop_ratio_med$|hop_ratio_abs$|kept_std$|amt_z_)")
N_FOLDS = 5
SEED = 0


def episode_folds(txs, n_folds, seed):
    """Fold per test row, grouped by connected components of payments sharing a wallet within 300s."""
    n = len(txs)
    codes = pd.factorize(np.r_[txs["sender"].to_numpy(), txs["receiver"].to_numpy()])[0]
    ts = txs["ts"].to_numpy()
    comp = _components(codes, np.r_[ts, ts], np.r_[np.arange(n), np.arange(n)], n, 300)
    perm = np.random.default_rng(seed).permutation(comp.max() + 1)
    return perm[comp] % n_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base")
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--bottom", type=float, default=0.8)
    ap.add_argument("--weight", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    base_path = f"submissions/{args.base}.csv"
    if not os.path.exists(base_path):
        subprocess.run([sys.executable, "scripts/realmotif_run.py", "--tag", args.base, "--seeds", "2"], check=True)

    train_txs, X = build_split("train", MODULES)
    test_txs, X_test = build_split("test", MODULES)
    keep = [c for c in X.columns if not DROP.search(c)]
    print(f"dropped {X.shape[1] - len(keep)} amount-evenness features, kept {len(keep)}", flush=True)
    X, X_test = X[keep].reset_index(drop=True), X_test[keep].reset_index(drop=True)
    y = load_labels().set_index("tx").loc[train_txs["tx"], "label"].to_numpy()

    base = pd.read_csv(base_path).set_index("TRANSACTION_HASH")["score"].loc[test_txs["tx"]].to_numpy()
    rank = pd.Series(base).rank(ascending=False, method="first").to_numpy()
    pseudo = np.full(len(base), -1)
    pseudo[rank <= args.top] = 1
    pseudo[rank > len(base) * (1 - args.bottom)] = 0
    print(f"pseudo-labels: {np.sum(pseudo == 1)} positive, {np.sum(pseudo == 0)} negative, "
          f"{np.sum(pseudo == -1)} unused", flush=True)

    fold = episode_folds(test_txs, N_FOLDS, SEED)
    raw = np.zeros(len(X_test))
    for k in range(N_FOLDS):
        use = (fold != k) & (pseudo >= 0)
        Xf = pd.concat([X, X_test[use]], ignore_index=True)
        yf = np.r_[y, pseudo[use]]
        wf = np.r_[np.ones(len(y)), np.full(use.sum(), args.weight)]
        hold = fold == k
        for seed in range(args.seeds):
            model = lgb.LGBMClassifier(**{**PARAMS, "random_state": SEED + seed})
            model.fit(Xf, yf, sample_weight=wf)
            raw[hold] += model.predict_proba(X_test[hold])[:, 1] / args.seeds
        print(f"fold {k}: {hold.sum()} test rows scored, {use.sum()} pseudo rows in training", flush=True)

    top = pd.Series(raw).rank(ascending=False).to_numpy() <= len(raw) // 100
    base_top = rank <= len(raw) // 100
    print(f"rank correlation with {args.base}: {pd.Series(raw).corr(pd.Series(base), method='spearman'):.3f}, "
          f"top 1% overlap {np.sum(top & base_top)} of {top.sum()}", flush=True)
    write("pseudo", test_txs, raw, base)


if __name__ == "__main__":
    main()
