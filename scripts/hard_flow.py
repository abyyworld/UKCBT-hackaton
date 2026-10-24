"""Week-long aggregate flow and counterparty history features for relay-like payments (tag: flow).

    .venv/bin/python -W ignore scripts/hard_flow.py [--explore]

Per payment: how much of the value the receiver gets leaves again within 1/6/24/72 h, how much the
sender received just before, day-by-day counterparty churn (Jaccard, new counterparties), pair history
(days active, first seen), wallet lifetime and cumulative net balance. A relay-only LightGBM is trained
on aggregate_relay + distributed_relay rows with folds grouped by cohort_id; labels and ids are used
only as targets and for grouping.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.getcwd())
from src.load import load_labels, load_split  # noqa: E402
from src.predict import write  # noqa: E402

CACHE = os.environ.get("FLOW_CACHE", "data/cache")
RELAY = ("aggregate_relay", "distributed_relay")
H = 3600
WINDOWS = {"1h": H, "6h": 6 * H, "24h": 24 * H, "72h": 72 * H}
BIG = np.int64(10 ** 11)


def txs_of(split):
    path = f"{CACHE}/flow_txs_{split}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    txs, _ = load_split(split)
    cols = ["tx", "ts", "sender", "receiver", "usd_in", "usd_out", "asset_in", "asset_out", "is_conversion"]
    txs = txs[[c for c in cols if c in txs.columns]].reset_index(drop=True)
    txs.to_parquet(path)
    return txs


class Range:
    """Sum / count of events per wallet within a time range, via a global sorted key and cumsum."""

    def __init__(self, wallet, ts, val):
        key = wallet.astype(np.int64) * BIG + ts.astype(np.int64)
        o = np.argsort(key, kind="stable")
        self.key = key[o]
        self.cum = np.concatenate([[0.0], np.cumsum(val[o])])

    def query(self, wallet, lo, hi):  # (lo, hi]
        base = wallet.astype(np.int64) * BIG
        a = np.searchsorted(self.key, base + lo, side="right")
        b = np.searchsorted(self.key, base + hi, side="right")
        return self.cum[b] - self.cum[a], (b - a).astype(float)


def features(txs):
    t = txs["ts"].to_numpy().astype(np.int64)
    t = t - t.min()
    v = np.nan_to_num(txs["usd_in"].to_numpy().astype(float))
    vo = np.nan_to_num(txs["usd_out"].to_numpy().astype(float)) if "usd_out" in txs else v
    codes, uniq = pd.factorize(pd.concat([txs["sender"], txs["receiver"]], ignore_index=True))
    s = codes[: len(txs)]
    r = codes[len(txs):]
    f = {}
    v1 = np.maximum(v, 1e-6)
    out_idx = Range(s, t, v)      # outgoing events by sender
    in_idx = Range(r, t, vo)      # incoming events by receiver
    for name, w in WINDOWS.items():
        so, sc = out_idx.query(r, t, t + w)          # receiver forwards after this payment
        f[f"r_fwd_{name}"] = so / v1
        f[f"r_fwdn_{name}"] = sc
        si, ci = in_idx.query(s, t - w - 1, t - 1)   # sender received just before
        f[f"s_pre_{name}"] = si / v1
        f[f"s_pren_{name}"] = ci
        ri, rc = in_idx.query(r, t - w - 1, t - 1)   # receiver received before (other inflow)
        f[f"r_prein_{name}"] = ri / v1
        ro, roc = out_idx.query(s, t, t + w)         # sender keeps sending after
        f[f"s_postout_{name}"] = ro / v1
        # causal asymmetry: a real relay hub receives first and forwards later
        rb, _ = out_idx.query(r, t - w - 1, t - 1)    # receiver sent before this payment
        sa, _ = in_idx.query(s, t, t + w)             # sender received after this payment
        f[f"r_fwd_asym_{name}"] = (so - rb) / np.maximum(so + rb, 1e-6)
        f[f"s_pre_asym_{name}"] = (si - sa) / np.maximum(si + sa, 1e-6)
        f[f"r_cons_{name}"] = np.minimum(so, v) / np.maximum(so, v1)
    # week-long net balance per wallet up to t (normalised by wallet's total volume)
    allw = np.concatenate([s, r])
    allt = np.concatenate([t, t])
    signed = np.concatenate([-v, vo])
    net = Range(allw, allt, signed)
    vol = Range(allw, allt, np.abs(signed))
    big_t = np.int64(10 ** 10)
    for side, w_ in (("s", s), ("r", r)):
        nb, _ = net.query(w_, -1, t - 1)
        tv, tn = vol.query(w_, -1, big_t)
        f[f"{side}_netbal"] = nb / np.maximum(tv, 1e-6)
        f[f"{side}_nevents"] = tn
        f[f"{side}_vol_rel"] = v / np.maximum(tv, 1e-6)
    # wallet lifetime and position
    ev = pd.DataFrame({"w": allw, "t": allt})
    g = ev.groupby("w")["t"]
    first, last = g.min(), g.max()
    for side, w_ in (("s", s), ("r", r)):
        fs = first.reindex(w_).to_numpy()
        ls = last.reindex(w_).to_numpy()
        f[f"{side}_age_h"] = (t - fs) / H
        f[f"{side}_left_h"] = (ls - t) / H
        f[f"{side}_span_h"] = (ls - fs) / H
    # pair history
    day = t // 86400
    pair = pd.Series(s.astype(np.int64) * BIG + r)
    pdf = pd.DataFrame({"p": pair, "t": t, "d": day})
    gp = pdf.groupby("p")
    f["pair_n"] = gp["t"].transform("size").to_numpy()
    f["pair_days"] = gp["d"].transform("nunique").to_numpy()
    f["pair_rank"] = gp["t"].rank(method="first").to_numpy() - 1
    f["pair_age_h"] = (t - gp["t"].transform("min").to_numpy()) / H
    f["pair_span_h"] = (gp["t"].transform("max").to_numpy() - gp["t"].transform("min").to_numpy()) / H
    rev = pd.Series(r.astype(np.int64) * BIG + s)
    rev_n = pair.value_counts()
    f["pair_rev_n"] = rev.map(rev_n).fillna(0).to_numpy()
    # day-by-day counterparty sets per wallet
    cp = pd.DataFrame({"w": allw, "c": np.concatenate([r, s]), "d": np.concatenate([day, day])}).drop_duplicates()
    nd = cp.groupby(["w", "d"]).size().rename("n")
    prev = cp.assign(d=cp["d"] + 1)
    inter = cp.merge(prev, on=["w", "d", "c"]).groupby(["w", "d"]).size().rename("i")
    dd = pd.concat([nd, inter], axis=1).fillna(0)
    dd["nprev"] = nd.reindex(pd.MultiIndex.from_arrays([dd.index.get_level_values(0), dd.index.get_level_values(1) - 1])).fillna(0).to_numpy()
    dd["jac"] = dd["i"] / np.maximum(dd["n"] + dd["nprev"] - dd["i"], 1)
    firstc = cp.groupby(["w", "c"])["d"].transform("min")
    cp["new"] = (cp["d"] == firstc).astype(float)
    cpd = cp.groupby(["w", "c"])["d"].transform("nunique")
    cp["oneday"] = (cpd == 1).astype(float)
    agg = cp.groupby(["w", "d"])[["new", "oneday"]].mean()
    dd = dd.join(agg)
    wd_days = cp.groupby("w")["d"].nunique()
    for side, w_ in (("s", s), ("r", r)):
        idx = pd.MultiIndex.from_arrays([w_, day])
        sub = dd.reindex(idx)
        f[f"{side}_day_ncp"] = sub["n"].to_numpy()
        f[f"{side}_day_jac"] = sub["jac"].to_numpy()
        f[f"{side}_day_newfrac"] = sub["new"].to_numpy()
        f[f"{side}_day_onedayfrac"] = sub["oneday"].to_numpy()
        f[f"{side}_ndays"] = wd_days.reindex(w_).to_numpy()
    # counterparty novelty for this pair from both ends
    pc = cp.set_index(["w", "c"])
    pdays = cp.groupby(["w", "c"])["d"].nunique()
    f["pair_cp_days"] = pdays.reindex(pd.MultiIndex.from_arrays([s, r])).to_numpy()
    # value-conserving onward hop: receiver sends 80-100% of v within 24 h in one payment
    o = np.lexsort((t, s))
    ss, st, sv = s[o], t[o], v[o]
    key = ss.astype(np.int64) * BIG + st
    a = np.searchsorted(key, r.astype(np.int64) * BIG + t, side="right")
    b = np.searchsorted(key, r.astype(np.int64) * BIG + t + 24 * H, side="right")
    best = np.zeros(len(t))
    nmatch = np.zeros(len(t))
    lens = b - a
    rows = np.repeat(np.arange(len(t)), np.minimum(lens, 50))
    offs = np.concatenate([np.arange(min(k, 50)) for k in lens]) if len(rows) else np.array([], int)
    if len(rows):
        rat = sv[a[rows] + offs] / v1[rows]
        good = (rat > 0.8) & (rat <= 1.0)
        np.add.at(nmatch, rows, good.astype(float))
        np.maximum.at(best, rows, np.where(rat <= 1.0, rat, 0))
    f["hop_n_conserve"] = nmatch
    f["hop_best_ratio"] = best
    del pc
    return pd.DataFrame(f).astype(float)


def main():
    explore = "--explore" in sys.argv
    tr = txs_of("train")
    lab = load_labels().set_index("tx").loc[tr["tx"]]
    X = features(tr)
    X.index = tr["tx"].to_numpy()
    relay = lab["family"].isin(RELAY).to_numpy()
    y = lab["label"].to_numpy()
    hard = (lab["difficulty"] == "hard").to_numpy()
    yr = (y[relay] == 1) & hard[relay]
    print("relay rows", relay.sum(), "hard pos", yr.sum())
    if explore:
        res = []
        for c in X.columns:
            x = X[c].to_numpy()[relay]
            x = np.nan_to_num(x, nan=-1e9)
            auc = roc_auc_score(yr, x)
            ap = max(average_precision_score(yr, x), average_precision_score(yr, -x))
            res.append((c, auc, ap))
        res = pd.DataFrame(res, columns=["f", "auc", "ap"]).assign(d=lambda d: (d.auc - 0.5).abs())
        print(res.sort_values("d", ascending=False).to_string())
    Xr = X[relay]
    groups = lab["cohort_id"].to_numpy()[relay]
    params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=300, verbose=-1, n_jobs=2)
    oof = np.zeros(len(Xr))
    for k, (a, b) in enumerate(GroupKFold(n_splits=6).split(Xr, yr, groups)):
        m = lgb.LGBMClassifier(**params, random_state=k)
        m.fit(Xr.iloc[a], yr[a])
        oof[b] = m.predict_proba(Xr.iloc[b])[:, 1]
    print(f"OOF within relay: AUC {roc_auc_score(yr, oof):.4f} AP {average_precision_score(yr, oof):.4f} base {yr.mean():.4f}")
    order = np.argsort(-oof)
    for k in (50, 100, 200):
        print(f"hard pos in top {k}: {int(yr[order[:k]].sum())}")
    if explore:
        return
    # honest train scores: relay rows get OOF, others get a full-relay model score shifted below
    full = lgb.LGBMClassifier(**params, random_state=0).fit(Xr, yr)
    tr_score = full.predict_proba(X)[:, 1]
    tr_score[relay] = oof
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": tr["tx"].to_numpy(), "score": tr_score}).to_parquet("data/oof/hard_flow.parquet")
    te = txs_of("test")
    Xt = features(te)
    raw = full.predict_proba(Xt[Xr.columns])[:, 1]
    print("test rows scoring > 0.5:", int((raw > 0.5).sum()), "> 0.2:", int((raw > 0.2).sum()))
    write("hard_flow", te, raw, Xt["r_fwd_24h"].rank().to_numpy())


if __name__ == "__main__":
    main()
