"""Group-level classifier: rebuild motifs as groups of first-time-pair payments, describe each whole group, and
learn which groups are laundering. Each payment then gets its group's out-of-fold score.

    python -m explore.groups

Groups: first-time-pair payments linked when they share a wallet within T seconds. Group labels (training only):
a group is positive when most of its payments are suspicious. Folds are grouped by scenario like src.cv.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.features.episodes import _components
from src.load import load_labels, load_split

TS = [300, 600]


def novelty(txs):
    ts = txs["ts"].to_numpy().astype(float)
    order = np.argsort(ts, kind="stable")
    nov = np.zeros(len(txs), bool)
    nov[order] = ~pd.Series(list(zip(txs["sender"], txs["receiver"]))).iloc[order].duplicated().to_numpy()
    return nov


def group_table(txs, T):
    """One row per group of linked first-time-pair payments, plus the group id of every payment (-1 if none)."""
    n = len(txs)
    ts = txs["ts"].to_numpy().astype(float)
    snd, rcv = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    nov = novelty(txs)
    nv = np.flatnonzero(nov)
    comp = _components(np.r_[snd[nv], rcv[nv]], np.r_[ts[nv], ts[nv]], np.r_[nv, nv], n, T)
    gid = np.where(nov, comp, -1)

    busy = pd.concat([txs["sender"], txs["receiver"]]).value_counts()
    d = pd.DataFrame({
        "g": gid, "ts": ts, "s": snd, "r": rcv, "usd_in": txs["usd_in"].to_numpy(),
        "usd_out": txs["usd_out"].to_numpy(), "ain": txs["asset_in"].to_numpy(), "aout": txs["asset_out"].to_numpy(),
        "conv": txs["is_conversion"].to_numpy().astype(float), "legs": txs["n_legs"].to_numpy().astype(float),
        "xrp_in": (txs["asset_in"].to_numpy() == "XRP").astype(float),
        "xrp_out": (txs["asset_out"].to_numpy() == "XRP").astype(float),
        "sbusy": busy.reindex(snd).to_numpy(), "rbusy": busy.reindex(rcv).to_numpy(),
    })
    d = d[d["g"] >= 0].sort_values(["g", "ts"])
    d["gap"] = d.groupby("g")["ts"].diff()
    g = d.groupby("g")
    cv = lambda x: x.std() / x.mean() if len(x) > 1 and x.mean() > 0 else 0.0
    st = pd.DataFrame({
        "size": g.size(), "span": g["ts"].max() - g["ts"].min(),
        "gap_med": g["gap"].median(), "gap_max": g["gap"].max(), "gap_cv": g["gap"].agg(cv),
        "n_snd": g["s"].nunique(), "n_rcv": g["r"].nunique(),
        "n_ain": g["ain"].nunique(), "n_aout": g["aout"].nunique(),
        "conv": g["conv"].mean(), "legs": g["legs"].mean(), "xrp_in": g["xrp_in"].mean(), "xrp_out": g["xrp_out"].mean(),
        "usd_med": g["usd_in"].median(), "usd_sum": g["usd_in"].sum(), "usd_cv": g["usd_in"].agg(cv),
        "busy_med": (g["sbusy"].median() + g["rbusy"].median()) / 2, "busy_min": g[["sbusy", "rbusy"]].min().min(axis=1),
    })
    st["n_wal"] = d.groupby("g").apply(lambda x: len(set(x["s"]) | set(x["r"])), include_groups=False)
    st["density"] = st["size"] / st["n_wal"]
    # value conservation through wallets that both receive and send inside the group
    inflow = d.groupby(["g", "r"])["usd_out"].sum()
    outflow = d.groupby(["g", "s"])["usd_in"].sum()
    both = inflow.index.intersection(outflow.index)
    kept = (outflow.loc[both] / inflow.loc[both]).clip(0, 3)
    st["n_pass"] = kept.groupby(level=0).size().reindex(st.index).fillna(0)
    st["kept_med"] = kept.groupby(level=0).median().reindex(st.index)
    st["kept_std"] = kept.groupby(level=0).std().reindex(st.index)
    srcs = set(d["s"]) - set(d["r"])
    st["e2e"] = (d[~d["r"].isin(set(d["s"]))].groupby("g")["usd_out"].sum()
                 / d[d["s"].isin(srcs)].groupby("g")["usd_in"].sum()).reindex(st.index)
    st["max_out"] = d.groupby(["g", "s"]).size().groupby(level=0).max()
    st["max_in"] = d.groupby(["g", "r"]).size().groupby(level=0).max()
    for c in ("span", "gap_med", "gap_max", "usd_med", "usd_sum", "busy_med", "busy_min"):
        st[c] = np.log1p(st[c])
    return st, gid


def oof_group_scores(txs, lab, T, seed=0):
    st, gid = group_table(txs, T)
    y_tx = lab["label"].to_numpy()
    sid = lab["scenario_id"].to_numpy()
    d = pd.DataFrame({"g": gid, "y": y_tx, "sid": sid})[gid >= 0]
    gy = (d.groupby("g")["y"].mean() > 0.5).astype(int).reindex(st.index)
    gs = d.groupby("g")["sid"].agg(lambda x: x.dropna().mode().iloc[0] if x.notna().any() else None).reindex(st.index)
    groups = gs.fillna(pd.Series(st.index.astype(str), index=st.index)).to_numpy()
    X, y = st, gy.to_numpy()
    oof = np.zeros(len(y))
    params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=10, n_estimators=500,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=4, random_state=seed)
    for fit, hold in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        m = lgb.LGBMClassifier(**params).fit(X.iloc[fit], y[fit])
        oof[hold] = m.predict_proba(X.iloc[hold])[:, 1]
    per_tx = np.full(len(txs), np.nan)
    ok = gid >= 0
    per_tx[ok] = pd.Series(oof, index=st.index).reindex(gid[ok]).to_numpy()
    return per_tx, st, y, oof


if __name__ == "__main__":
    txs, _ = load_split("train")
    lab = load_labels().set_index("tx").loc[txs["tx"]].reset_index()
    y = lab["label"].to_numpy()
    base = pd.read_parquet("data/oof/final.parquet").set_index("tx")["oof"].reindex(lab["tx"]).to_numpy()
    print(f"v13 OOF AP {100 * average_precision_score(y, base):.2f}")
    for T in TS:
        per_tx, st, gy, goof = oof_group_scores(txs, lab, T)
        big = st["size"].to_numpy() >= 3
        print(f"T={T}: groups {len(st)}, positive {gy.sum()}, group AUC {roc_auc_score(gy, goof):.3f}, "
              f"AUC size>=3 {roc_auc_score(gy[big], goof[big]):.3f}")
        g = np.nan_to_num(per_tx, nan=0.0)
        rb = rankdata(base) / len(base)
        for w in (0.25, 0.5, 1.0):
            print(f"   base + {w} * group: {100 * average_precision_score(y, base + w * g):.2f}   "
                  f"base * (1 + {w} * group): {100 * average_precision_score(y, base * (1 + w * g)):.2f}   "
                  f"rank mix: {100 * average_precision_score(y, rb + w * g):.2f}")
