"""Scheme-level model: score candidate schemes, then score transactions from their schemes.

    python scripts/scheme_model.py

1. For every payment (the anchor) build a candidate scheme at 13 scales: all payments that share a wallet with
   the anchor, or with a payment that does (two hops), inside a time window around the anchor. Two hops: all payments
   within 60s, 300s, 900s; first-time wallet pairs only within 30m, 1h, 2h. One hop: first-time pairs within 1h, 2h,
   3h; all payments within 300s, 900s, 30m, 1h.
2. Describe each scheme: size, span, gaps, wallets, in and out degrees, pass-through wallets, value kept per hop,
   amount drift, conversions, assets, isolation (the wallets' other activity during the scheme), new wallets, hour.
3. Train a LightGBM scheme classifier per scale (target: the anchor's label), grouped folds by scenario.
4. Each transaction gets its own scheme score plus the max and mean score of every scheme it belongs to, per scale.
   A small transaction-level LightGBM on those scores gives the final score.

Labels, family, difficulty and scenario_id are used only as targets and for grouping folds.
Writes data/oof/scheme.parquet (tx, oof) and submissions/scheme.csv.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import report
from src.load import load_labels, load_split
from src.predict import write

SEED = 0
N_FOLDS = 5
SCALES = [("all60", False, 60), ("all300", False, 300), ("all900", False, 900),
          ("first30m", True, 1800), ("first1h", True, 3600), ("first2h", True, 7200),
          # one hop only: the anchor's own wallets, purer for slow schemes
          ("h1_first1h", True, 3600, 1), ("h1_first2h", True, 7200, 1), ("h1_first3h", True, 10800, 1),
          ("h1_all300", False, 300, 1), ("h1_all900", False, 900, 1), ("h1_all1800", False, 1800, 1),
          ("h1_all3600", False, 3600, 1)]
SCHEME_PARAMS = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=31, min_child_samples=30,
                     feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                     n_estimators=400, verbose=-1, n_jobs=4)
TX_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_child_samples=80,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 n_estimators=300, verbose=-1, n_jobs=4)
TX_SEEDS = 3
# amount evenness inside a scheme differs between train and test (test schemes are much more uneven),
# so these features would not transfer; tempo, structure, isolation and value balance stay
UNEVEN = ["lusd_std", "usd_maxmin", "drift", "keep_med", "keep_min", "keep_max", "a_lusd_dev"]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def expand(lo, hi):
    """For ranges [lo, hi) return (range id, position) for every element."""
    n = hi - lo
    rid = np.repeat(np.arange(len(lo)), n)
    pos = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
    return rid, pos


def prepare(txs):
    t = txs.sort_values(["ts", "tx"]).reset_index(drop=True)
    wallets = pd.Index(pd.unique(np.r_[t["sender"].to_numpy(), t["receiver"].to_numpy()]))
    t["s"] = wallets.get_indexer(t["sender"])
    t["r"] = wallets.get_indexer(t["receiver"])
    pair = t["s"].astype(np.int64) * len(wallets) + t["r"]
    t["first"] = ~pair.duplicated()
    t["usd"] = t["usd_in"].fillna(t["usd_out"]).fillna(0.0)
    t["conv"] = t["is_conversion"].astype(float)
    t["ain"] = pd.factorize(t["asset_in"])[0]
    t["aout"] = pd.factorize(t["asset_out"])[0]
    # all wallet events (both roles) of the split, sorted by wallet then time, for isolation counts
    ev_w = np.r_[t["s"].to_numpy(), t["r"].to_numpy()]
    ev_t = np.r_[t["ts"].to_numpy(), t["ts"].to_numpy()]
    key = np.sort(ev_w.astype(np.int64) * 10**10 + ev_t)
    first_seen = pd.Series(ev_t).groupby(ev_w).min()
    last_seen = pd.Series(ev_t).groupby(ev_w).max()
    total = pd.Series(ev_t).groupby(ev_w).size()
    t.attrs.update(key=key, first_seen=first_seen, last_seen=last_seen, total=total)
    return t


def candidates(t, first_only, w, hops=2):
    """(anchor, member) pairs: payments within +-w of the anchor reachable within two wallet hops."""
    sub = np.where(t["first"].to_numpy())[0] if first_only else np.arange(len(t))
    ts = t["ts"].to_numpy()
    s, r = t["s"].to_numpy(), t["r"].to_numpy()
    ew = np.r_[s[sub], r[sub]].astype(np.int64)
    et = np.r_[ts[sub], ts[sub]]
    ei = np.r_[sub, sub]
    o = np.lexsort((et, ew))
    ew, et, ei = ew[o], et[o], ei[o]
    key = ew * 10**10 + et

    def hop(anchor, wallet):
        t_a = ts[anchor]
        lo = np.searchsorted(key, wallet * 10**10 + t_a - w, "left")
        hi = np.searchsorted(key, wallet * 10**10 + t_a + w, "right")
        rid, pos = expand(lo, hi)
        return anchor[rid], ei[pos]

    anchors = sub
    a1, m1 = hop(np.r_[anchors, anchors], np.r_[s[anchors], r[anchors]].astype(np.int64))
    one = pd.DataFrame({"a": np.r_[anchors, a1], "m": np.r_[anchors, m1]}).drop_duplicates()
    if hops == 1:
        return one.sort_values(["a", "m"]).reset_index(drop=True)
    a, m = one["a"].to_numpy(), one["m"].to_numpy()
    a2, m2 = hop(np.r_[a, a], np.r_[s[m], r[m]].astype(np.int64))
    pairs = pd.DataFrame({"a": np.r_[a, a2], "m": np.r_[m, m2]}).drop_duplicates()
    return pairs.sort_values(["a", "m"]).reset_index(drop=True)


def scheme_features(t, pairs):
    a = pairs["a"].to_numpy()
    m = pairs["m"].to_numpy()
    ts = t["ts"].to_numpy()[m].astype(float)
    usd = t["usd"].to_numpy()[m]
    d = pd.DataFrame({"a": a, "ts": ts, "usd": usd, "lusd": np.log1p(usd), "conv": t["conv"].to_numpy()[m],
                      "first": t["first"].to_numpy()[m].astype(float), "ain": t["ain"].to_numpy()[m],
                      "aout": t["aout"].to_numpy()[m], "s": t["s"].to_numpy()[m], "r": t["r"].to_numpy()[m],
                      "tag": t["dest_tag"].notna().to_numpy()[m].astype(float),
                      "memo": t["has_memo"].to_numpy()[m].astype(float)})
    g = d.groupby("a", sort=True)
    f = pd.DataFrame(index=g.size().index)
    f["n"] = g.size()
    f["t0"], f["t1"] = g["ts"].min(), g["ts"].max()
    f["span"] = f["t1"] - f["t0"]
    for c in ("conv", "first", "tag", "memo"):
        f[f"{c}_share"] = g[c].mean()
    f["usd_mean"] = g["usd"].mean()
    f["usd_sum"] = g["usd"].sum()
    f["lusd_std"] = g["lusd"].std()
    f["usd_maxmin"] = np.log1p(g["usd"].max()) - np.log1p(g["usd"].min())
    f["n_ain"], f["n_aout"] = g["ain"].nunique(), g["aout"].nunique()
    f["n_pairs"] = d.drop_duplicates(["a", "s", "r"]).groupby("a").size()
    f["repeat_pairs"] = f["n"] - f["n_pairs"]
    # drift: correlation of log amount with time inside the scheme
    d["tt"] = d["ts"] - d["a"].map(f["t0"]).to_numpy()
    d["tl"] = d["tt"] * d["lusd"]
    d["t2"] = d["tt"] ** 2
    d["l2"] = d["lusd"] ** 2
    sm = d.groupby("a")[["tt", "lusd", "tl", "t2", "l2"]].sum()
    n = f["n"]
    cov = sm["tl"] / n - sm["tt"] * sm["lusd"] / n**2
    vt = sm["t2"] / n - (sm["tt"] / n) ** 2
    vl = sm["l2"] / n - (sm["lusd"] / n) ** 2
    f["drift"] = (cov / np.sqrt(vt * vl + 1e-12)).where((vt > 0) & (vl > 1e-9), 0.0)
    # gaps between consecutive payments
    order = np.lexsort((ts, a))
    gs = np.diff(ts[order])
    gs = pd.Series(np.where(a[order][1:] == a[order][:-1], gs, np.nan)).groupby(a[order][1:])
    f["gap_mean"], f["gap_max"], f["gap_min"], f["gap_std"] = gs.mean(), gs.max(), gs.min(), gs.std()
    f["gap_med"] = gs.median()
    # wallet roles inside the scheme
    out = d.groupby(["a", "s"]).agg(out_n=("usd", "size"), out_usd=("usd", "sum"), out_k=("r", "nunique"),
                                    out_t0=("ts", "min"), out_t1=("ts", "max"))
    inn = d.groupby(["a", "r"]).agg(in_n=("usd", "size"), in_usd=("usd", "sum"), in_k=("s", "nunique"),
                                    in_t0=("ts", "min"), in_t1=("ts", "max"))
    out.index.names = inn.index.names = ["a", "w"]
    wr = out.join(inn, how="outer").fillna({"out_n": 0, "out_usd": 0, "out_k": 0, "in_n": 0, "in_usd": 0, "in_k": 0})
    wr = wr.reset_index()
    wr["pass"] = (wr["in_n"] > 0) & (wr["out_n"] > 0)
    wr["src"] = (wr["in_n"] == 0)
    wr["sink"] = (wr["out_n"] == 0)
    wr["keep"] = np.where(wr["pass"], wr["out_usd"] / (wr["in_usd"] + 1e-9), np.nan)
    wr["in_first"] = wr["pass"] & (wr["in_t0"] <= wr["out_t0"])
    wg = wr.groupby("a")
    f["n_wallets"] = wg.size()
    f["n_pass"], f["n_src"], f["n_sink"] = wg["pass"].sum(), wg["src"].sum(), wg["sink"].sum()
    f["pass_share"] = f["n_pass"] / f["n_wallets"]
    f["pass_in_first"] = wg["in_first"].mean()
    f["max_in"], f["max_out"] = wg["in_k"].max(), wg["out_k"].max()
    f["n_in2"] = (wr["in_k"] >= 2).groupby(wr["a"]).sum()
    f["n_out2"] = (wr["out_k"] >= 2).groupby(wr["a"]).sum()
    f["keep_med"], f["keep_min"], f["keep_max"] = wg["keep"].median(), wg["keep"].min(), wg["keep"].max()
    left = f["n_src"] + f["n_pass"]
    right = f["n_sink"] + f["n_pass"]
    f["density"] = f["n_pairs"] / (left * right).clip(lower=1)
    f["wallets_per_tx"] = f["n_wallets"] / f["n"]
    src_usd = wr["out_usd"].where(wr["src"], 0).groupby(wr["a"]).sum()
    sink_usd = wr["in_usd"].where(wr["sink"], 0).groupby(wr["a"]).sum()
    f["sink_over_src"] = np.log1p(sink_usd) - np.log1p(src_usd)
    # isolation: each wallet's activity in the split during the scheme span that is not part of the scheme
    key = t.attrs["key"]
    ww = wr["w"].to_numpy().astype(np.int64)
    w0 = wr["a"].map(f["t0"]).to_numpy().astype(np.int64)
    w1 = wr["a"].map(f["t1"]).to_numpy().astype(np.int64)
    pad = 1800
    during = (np.searchsorted(key, ww * 10**10 + w1, "right") - np.searchsorted(key, ww * 10**10 + w0, "left"))
    around = (np.searchsorted(key, ww * 10**10 + w1 + pad, "right")
              - np.searchsorted(key, ww * 10**10 + w0 - pad, "left"))
    inside = wr["in_n"].to_numpy() + wr["out_n"].to_numpy()
    wr["other"] = during - inside
    wr["other_pad"] = around - inside
    wr["iso"] = (wr["other"] == 0)
    wr["new"] = wr["w"].map(t.attrs["first_seen"]).to_numpy() >= w0 - pad
    wr["gone"] = wr["w"].map(t.attrs["last_seen"]).to_numpy() <= w1 + pad
    wr["frac_in"] = inside / wr["w"].map(t.attrs["total"]).to_numpy()
    wg = wr.groupby("a")
    f["other_sum"], f["other_max"] = wg["other"].sum(), wg["other"].max()
    f["other_pad_mean"] = wg["other_pad"].mean()
    f["iso_share"], f["new_share"], f["gone_share"] = wg["iso"].mean(), wg["new"].mean(), wg["gone"].mean()
    f["frac_in_mean"], f["frac_in_max"] = wg["frac_in"].mean(), wg["frac_in"].max()
    f["other_rate"] = f["other_sum"] / (f["span"] + 60) / f["n_wallets"]
    # the anchor inside its own scheme
    ia = f.index.to_numpy()
    f["a_pos"] = (ts_a := t["ts"].to_numpy()[ia]) - f["t0"]
    f["a_pos"] = f["a_pos"] / (f["span"] + 1)
    f["a_hour"] = (ts_a // 3600) % 24
    f["start_hour"] = (f["t0"] // 3600) % 24
    f["a_first"] = t["first"].to_numpy()[ia].astype(float)
    wri = wr.set_index(["a", "w"])
    for role, col in (("s", "s"), ("r", "r")):
        idx = pd.MultiIndex.from_arrays([ia, t[col].to_numpy()[ia]])
        sub = wri.reindex(idx)
        for c in ("in_k", "out_k", "other", "frac_in"):
            f[f"a{role}_{c}"] = sub[c].to_numpy()
    f["a_lusd_dev"] = np.log1p(t["usd"].to_numpy()[ia]) - g["lusd"].mean()
    return f.drop(columns=["t0", "t1"] + UNEVEN)


def build(name):
    txs, _ = load_split(name)
    t = prepare(txs)
    feats, members = {}, {}
    for scale, first_only, w, *hops in SCALES:
        pairs = candidates(t, first_only, w, *hops)
        feats[scale] = scheme_features(t, pairs)
        members[scale] = pairs
        log(f"{name} {scale}: {len(feats[scale])} schemes, {len(pairs)} memberships, "
            f"median size {feats[scale]['n'].median():.0f}")
    return t, feats, members


def tx_scores(t, f, pairs, score, prefix):
    """Own scheme score plus max and mean over every scheme the payment belongs to."""
    sc = pd.Series(score, index=f.index)
    p = pairs.assign(s=pairs["a"].map(sc).to_numpy())
    agg = p.groupby("m")["s"].agg(["max", "mean", "size"])
    out = pd.DataFrame(index=np.arange(len(t)))
    out[f"{prefix}_own"] = sc.reindex(out.index).to_numpy()
    out[f"{prefix}_max"] = agg["max"].reindex(out.index).to_numpy()
    out[f"{prefix}_mean"] = agg["mean"].reindex(out.index).to_numpy()
    out[f"{prefix}_cnt"] = agg["size"].reindex(out.index).to_numpy()
    out[f"{prefix}_n"] = f["n"].reindex(out.index).to_numpy()
    return out


def basic(t):
    total = t.attrs["total"]
    return pd.DataFrame({"lusd": np.log1p(t["usd"]), "conv": t["conv"], "first": t["first"].astype(float),
                         "legs": t["n_legs"], "hour": (t["ts"] // 3600) % 24,
                         "s_total": t["s"].map(total), "r_total": t["r"].map(total),
                         "tag": t["dest_tag"].notna().astype(float)})


def main():
    lab_all = load_labels().set_index("tx")
    tr, tr_f, tr_m = build("train")
    te, te_f, te_m = build("test")
    lab = lab_all.loc[tr["tx"]]
    y = lab["label"].to_numpy()
    groups = lab["scenario_id"].fillna(pd.Series(lab.index, index=lab.index)).to_numpy()
    fold = np.zeros(len(tr), int)
    for k, (_, hold) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
                                  .split(np.zeros(len(y)), y, groups)):
        fold[hold] = k

    # stage 1: scheme classifiers. For fold k the model never sees anchors of fold k, and fold-k payments
    # read scheme scores only from that model.
    tr_parts, te_parts = [], []
    for scale, *_ in SCALES:
        f, ft = tr_f[scale], te_f[scale][tr_f[scale].columns]
        ya, fa = y[f.index], fold[f.index]
        per_fold = []
        for k in range(N_FOLDS):
            mdl = lgb.LGBMRegressor(**SCHEME_PARAMS, random_state=SEED + k)
            mdl.fit(f[fa != k], ya[fa != k])
            per_fold.append(tx_scores(tr, f, tr_m[scale], mdl.predict(f), scale))
        merged = per_fold[0].copy()
        for k in range(N_FOLDS):
            merged.loc[fold == k] = per_fold[k].loc[fold == k]
        tr_parts.append(merged)
        mdl = lgb.LGBMRegressor(**SCHEME_PARAMS, random_state=SEED)
        mdl.fit(f, ya)
        te_parts.append(tx_scores(te, ft, te_m[scale], mdl.predict(ft), scale))
        own = merged[f"{scale}_own"].fillna(0).to_numpy()
        report(own, lab, f"scheme {scale} own")

    # basic transaction features (basic()) lowered OOF AP by about 0.7, so only scheme scores go in
    X = pd.concat(tr_parts, axis=1)
    Xt = pd.concat(te_parts, axis=1)[X.columns]
    for Z in (X, Xt):
        own = Z[[c for c in Z.columns if c.endswith("_own")]]
        mx = Z[[c for c in Z.columns if c.endswith("_max")]]
        Z["own_max"], Z["own_mean"] = own.max(axis=1), own.mean(axis=1)
        Z["max_max"], Z["max_mean"] = mx.max(axis=1), mx.mean(axis=1)

    # stage 2: transaction model on scheme scores, same folds
    oof = np.zeros(len(y))
    for seed in range(TX_SEEDS):
        for k in range(N_FOLDS):
            mdl = lgb.LGBMClassifier(**TX_PARAMS, random_state=SEED + 10 * seed + k)
            mdl.fit(X[fold != k], y[fold != k])
            oof[fold == k] += mdl.predict_proba(X[fold == k])[:, 1] / TX_SEEDS
    report(X["own_max"].to_numpy(), lab, "scheme max of own scores")
    report(oof, lab, "scheme model OOF")
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": tr["tx"].to_numpy(), "oof": oof}).to_parquet("data/oof/scheme.parquet")

    raw = np.zeros(len(te))
    for seed in range(TX_SEEDS):
        mdl = lgb.LGBMClassifier(**TX_PARAMS, random_state=SEED + seed)
        mdl.fit(X, y)
        raw += mdl.predict_proba(Xt)[:, 1] / TX_SEEDS
    write("scheme", te, raw, Xt["max_mean"].fillna(0).to_numpy())
    log("done")


if __name__ == "__main__":
    main()
