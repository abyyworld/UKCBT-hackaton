"""Stage 2: collective scoring. Each transaction gets the stage-1 scores of the transactions it is linked to
(same episode at several scales, same wallets within time windows), then a second model is trained on top.

    python -m src.stage2 --tag v3 [--modules local,episodes] [--predict]
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import PARAMS, report, training_frame
from src.features import build_split
from src.features.episodes import SCALES, _components
from src.predict import fit_predict, write

NEIGH_WINDOWS = [300, 3600, 21600]


def _wallet_window_mean(wallets, times, scores, W):
    """Mean score of the other transactions on the same wallet within +-W seconds."""
    order = np.lexsort((times, wallets))
    w, t, sc = wallets[order], times[order], scores[order]
    out_sum, out_cnt = np.zeros(len(w)), np.zeros(len(w))
    starts = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1]
    for a, b in zip(starts, np.r_[starts[1:], len(w)]):
        tt, cs = t[a:b], np.r_[0.0, np.cumsum(sc[a:b])]
        hi = np.searchsorted(tt, tt + W, "right")
        lo = np.searchsorted(tt, tt - W, "left")
        out_sum[a:b] = cs[hi] - cs[lo] - sc[a:b]
        out_cnt[a:b] = hi - lo - 1
    res_sum, res_cnt = np.empty(len(w)), np.empty(len(w))
    res_sum[order], res_cnt[order] = out_sum, out_cnt
    return res_sum, res_cnt


MOTIF_FAMILIES = {"bipartite", "chain", "cycle", "fan_in", "fan_out", "fan_out_in", "multi_stage", "random", "stack"}
RELAY_FAMILIES = {"aggregate_relay", "distributed_relay"}


def aux_targets(lab):
    """Training targets. Family is used only as a target to learn from, never as a feature."""
    return {
        "susp": lab["label"].to_numpy(),
        "motif": lab["family"].isin(MOTIF_FAMILIES).to_numpy().astype(int),
        "relay": lab["family"].isin(RELAY_FAMILIES).to_numpy().astype(int),
    }


def collective_features(txs, score, prefix=""):
    txs = txs.reset_index(drop=True)
    n = len(txs)
    t = txs["ts"].to_numpy().astype(float)
    s, r = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    pair = pd.Series(list(zip(s, r)))
    pair_new = (pair.map(pair.value_counts()) == 1).to_numpy()
    idx = np.arange(n)
    f = pd.DataFrame(index=txs["tx"].to_numpy())
    f["s1"] = score
    for kind, T in [sc for sc in SCALES if sc[1] in (90, 300, 600, 1800, 3600, 7200)]:
        keep = np.ones(n, bool) if kind == "all" else pair_new
        ep = _components(np.r_[s[keep], r[keep]], np.r_[t[keep], t[keep]], np.r_[idx[keep], idx[keep]], n, T)
        if kind != "all":
            ep = np.where(keep, ep, -1 - idx)
        g = pd.Series(score).groupby(ep)
        tot, cnt, mx = g.transform("sum").to_numpy(), g.transform("size").to_numpy(), g.transform("max").to_numpy()
        others = np.where(cnt > 1, (tot - score) / np.maximum(cnt - 1, 1), np.nan)
        f[f"ep_{kind}{T}_mean_others"] = others
        f[f"ep_{kind}{T}_max"] = mx
        f[f"ep_{kind}{T}_mean_all"] = tot / cnt
    for W in NEIGH_WINDOWS:
        ss, sc = _wallet_window_mean(np.r_[s, r], np.r_[t, t], np.r_[score, score], W)
        s_sum, s_cnt, r_sum, r_cnt = ss[:n], sc[:n], ss[n:], sc[n:]
        f[f"nb_s_{W}"] = np.where(s_cnt > 0, s_sum / np.maximum(s_cnt, 1), np.nan)
        f[f"nb_r_{W}"] = np.where(r_cnt > 0, r_sum / np.maximum(r_cnt, 1), np.nan)
        f[f"nb_both_{W}"] = (s_sum + r_sum) / np.maximum(s_cnt + r_cnt, 1)
    return f.add_prefix(prefix)



SUBSETS = {"realmotif": MOTIF_FAMILIES, "realrelay": RELAY_FAMILIES}


def oof_and_test(X, y, groups, seed=0, mask=None, X_test=None, n_splits=5):
    """Out-of-fold scores, same folds as src.cv.oof_scores, plus, when X_test is given, the mean of the five fold
    models on the test rows. With `mask`, each fold model is fit only on its training rows inside the mask (for
    example all motif-shaped payments, real and look-alike), so it spends its capacity on telling the planted ones
    from the look-alikes; the mask comes from family and only picks training rows, it is never a feature.
    Fold-model test scores have the same spread as the OOF scores the next stage was trained on, unlike a fit on
    all of train, which is sharper."""
    oof, test = np.zeros(len(y)), np.zeros(0 if X_test is None else len(X_test))
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit, hold in folds.split(X, y, groups):
        if mask is not None:
            fit = fit[mask[fit]]
        model = lgb.LGBMClassifier(**{**PARAMS, "random_state": seed})
        model.fit(X.iloc[fit], y[fit])
        oof[hold] = model.predict_proba(X.iloc[hold])[:, 1]
        if X_test is not None:
            test += model.predict_proba(X_test[X.columns])[:, 1] / n_splits
    return oof, test


def fold_models_scores(X, y, groups, seed=0, mask=None, X_test=None, n_splits=5):
    """Score every train row with each of the five fold models. Returns (P, fold, test): P[:, k] is fold model k on
    all rows, fold[i] the fold row i is held out in, test the mean of the fold models on X_test (or None)."""
    P, fold = np.zeros((len(y), n_splits)), np.zeros(len(y), int)
    test = None if X_test is None else np.zeros(len(X_test))
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (fit, hold) in enumerate(folds.split(X, y, groups)):
        if mask is not None:
            fit = fit[mask[fit]]
        model = lgb.LGBMClassifier(**{**PARAMS, "random_state": seed})
        model.fit(X.iloc[fit], y[fit])
        P[:, k] = model.predict_proba(X)[:, 1]
        fold[hold] = k
        if X_test is not None:
            test += model.predict_proba(X_test[X.columns])[:, 1] / n_splits
    return P, fold, test


def consistent_collective(txs, P, fold, prefix=""):
    """Collective features where row i's neighbours are all scored by the fold model that never saw i's fold.
    Plain OOF scores leak: a normal payment on a planted wallet sits in another fold, so its OOF score comes from a
    model that learned the planted payments next to it, and stage 2 learns to trust that leaked neighbour score."""
    out = None
    for k in range(P.shape[1]):
        f = collective_features(txs, P[:, k], prefix)
        if out is None:
            out = f.copy()
        rows = fold == k
        out.iloc[rows] = f.iloc[rows].to_numpy()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--modules", default="local,episodes")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--aux", action="store_true", help="also stack motif and relay membership models")
    ap.add_argument("--sub", default="realmotif,realrelay",
                    help="subset models whose scores also feed stage 2 (comma list of SUBSETS keys, or none)")
    ap.add_argument("--rounds", type=int, default=1, help="rounds of collective scoring on top of stage 1")
    ap.add_argument("--consistent", action="store_true",
                    help="fold-consistent collective features (no neighbour-score leak); stage 2 then uses one "
                         "stage-1 fold split per seed")
    ap.add_argument("--testpred", default="full", choices=["full", "folds"],
                    help="score test with models fit on all of train (full) or with the mean of the fold models")
    args = ap.parse_args()
    modules = args.modules.split(",")

    X, lab, groups = training_frame(modules)
    y = lab["label"].to_numpy()
    train_txs, _ = build_split("train", modules)
    test_txs, X_test = build_split("test", modules) if args.predict else (None, None)
    folds_mode = args.predict and args.testpred == "folds"

    def run(Xtr, target, mask=None, Xte=None):
        """Seed-averaged OOF scores, and test scores: fold models in folds mode, else a fit on all of train."""
        res = [oof_and_test(Xtr, target, groups, sd, mask, Xte if folds_mode else None) for sd in range(args.seeds)]
        oof = np.mean([r[0] for r in res], axis=0)
        if Xte is None:
            return oof, None
        if folds_mode:
            return oof, np.mean([r[1] for r in res], axis=0)
        keep = np.ones(len(target), bool) if mask is None else mask
        return oof, fit_predict(Xtr[keep], target[keep], Xte, args.seeds)

    targets = aux_targets(lab) if args.aux else {"susp": y}
    subs_c = [k for k in args.sub.split(",") if k in SUBSETS]
    if args.consistent:
        masks_c = {k: None for k in targets} | {k: lab["family"].isin(SUBSETS[k]).to_numpy() for k in subs_c}
        tgt = {**targets, **{k: y for k in subs_c}}
        oof2, raw, s1_test = np.zeros(len(y)), 0.0, None
        for sd in range(args.seeds):
            parts, parts_t = [X], [X_test]
            for k, m in masks_c.items():
                P, fold, te = fold_models_scores(X, tgt[k], groups, sd, m, X_test)
                if k == "susp":
                    s1_oof = P[np.arange(len(y)), fold]
                    report(s1_oof, lab, f"{args.tag} stage1 seed{sd}")
                    s1_test = te
                parts.append(consistent_collective(train_txs, P, fold, f"{k}_"))
                if args.predict:
                    if args.testpred == "full":
                        keep = np.ones(len(y), bool) if m is None else m
                        te = fit_predict(X[keep], tgt[k][keep], X_test, 1)
                    parts_t.append(collective_features(test_txs, te, f"{k}_"))
            X2 = pd.concat(parts, axis=1)
            o, _ = oof_and_test(X2, y, groups, sd)
            oof2 += o / args.seeds
            report(o, lab, f"{args.tag} stage2 seed{sd}")
            if args.predict:
                raw = raw + fit_predict(X2, y, pd.concat(parts_t, axis=1), 1) / args.seeds
        report(oof2, lab, f"{args.tag} stage2")
        os.makedirs("data/oof", exist_ok=True)
        pd.DataFrame({"tx": lab.index, "oof": oof2}).to_parquet(f"data/oof/{args.tag}.parquet")
        if args.predict:
            write(args.tag, test_txs, raw, s1_test)
        raise SystemExit
    subs = [k for k in args.sub.split(",") if k in SUBSETS]
    masks = {k: lab["family"].isin(SUBSETS[k]).to_numpy() for k in subs}
    oof1, test1 = {}, {}
    for k, t in targets.items():
        oof1[k], test1[k] = run(X, t, None, X_test)
    report(oof1["susp"], lab, f"{args.tag} stage1")
    for k in subs:
        oof1[k], test1[k] = run(X, y, masks[k], X_test)

    def stack(X_base, txs, scores, prefix_round=""):
        C = pd.concat([collective_features(txs, o, f"{prefix_round}{k}_") for k, o in scores.items()], axis=1)
        return pd.concat([X_base, C], axis=1)

    X2 = stack(X, train_txs, oof1)
    X2_test = stack(X_test, test_txs, test1) if args.predict else None
    oof2, raw = run(X2, y, None, X2_test)
    report(oof2, lab, f"{args.tag} stage2")
    for k in range(2, args.rounds + 1):
        X2 = stack(X2, train_txs, {"": oof2}, f"r{k}")
        X2_test = stack(X2_test, test_txs, {"": raw}, f"r{k}") if args.predict else None
        oof2, raw = run(X2, y, None, X2_test)
        report(oof2, lab, f"{args.tag} round{k}")
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": oof2}).to_parquet(f"data/oof/{args.tag}.parquet")

    if args.predict:
        write(args.tag, test_txs, raw, test1["susp"])
