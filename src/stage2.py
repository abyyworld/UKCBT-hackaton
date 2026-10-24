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

from src.cv import PARAMS, oof_scores, report, training_frame
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--modules", default="local,episodes")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--aux", action="store_true", help="also stack motif and relay membership models")
    ap.add_argument("--rounds", type=int, default=1, help="rounds of collective scoring on top of stage 1")
    args = ap.parse_args()
    modules = args.modules.split(",")

    X, lab, groups = training_frame(modules)
    y = lab["label"].to_numpy()
    train_txs, _ = build_split("train", modules)
    targets = aux_targets(lab) if args.aux else {"susp": y}
    oof1 = {k: np.mean([oof_scores(X, t, groups, seed=sd) for sd in range(args.seeds)], axis=0) for k, t in targets.items()}
    report(oof1["susp"], lab, f"{args.tag} stage1")
    C = pd.concat([collective_features(train_txs, o, f"{k}_") for k, o in oof1.items()], axis=1)
    X2 = pd.concat([X, C], axis=1)
    oof2 = np.mean([oof_scores(X2, y, groups, seed=sd) for sd in range(args.seeds)], axis=0)
    report(oof2, lab, f"{args.tag} stage2")
    later = []  # (X_k, scores) for rounds after the first
    for k in range(2, args.rounds + 1):
        Xk = pd.concat([X2, collective_features(train_txs, oof2, f"r{k}_")], axis=1)
        oof2 = np.mean([oof_scores(Xk, y, groups, seed=sd) for sd in range(args.seeds)], axis=0)
        report(oof2, lab, f"{args.tag} round{k}")
        later.append(Xk)
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": oof2}).to_parquet(f"data/oof/{args.tag}.parquet")

    if args.predict:
        test_txs, X_test = build_split("test", modules)
        s1_test = {k: fit_predict(X, t, X_test, args.seeds) for k, t in targets.items()}
        C_test = pd.concat([collective_features(test_txs, o, f"{k}_") for k, o in s1_test.items()], axis=1)
        X2_test = pd.concat([X_test, C_test], axis=1)
        raw = fit_predict(X2, y, X2_test, args.seeds)
        for k, Xk in enumerate(later, start=2):
            Xk_test = pd.concat([X2_test, collective_features(test_txs, raw, f"r{k}_")], axis=1)
            raw = fit_predict(Xk, y, Xk_test, args.seeds)
        write(args.tag, test_txs, raw, s1_test["susp"])
