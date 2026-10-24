"""Odd-one-out relay units: score payments by how unusual their relay unit is on week-long stats.

Unit (no labels): a recurring spine pair A->B (>=3 payments on >=3 days); each spine payment pulls in
payments into A in the 6h before it and payments out of B in the 6h after it. Unit stats (lower = odder):
  ncp   mean distinct-counterparty count (whole week) of the unit's wallets
  reuse mean number of units the spine wallets A, B belong to
  back  originators' inflow in the 24h before they pay A, relative to what they pay
Unit score = mean percentile of (-ncp, -reuse, -back). Payment score = max over its units, 0 if none.
Writes data/oof/hard_odd.parquet (train, no fitting on labels) and submissions/hard_odd.csv.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.load import load_split, load_labels  # noqa: E402
from src.predict import write  # noqa: E402

WIN, BACK = 6 * 3600, 86400


def unit_scores(t):
    t = t.sort_values("ts").reset_index(drop=True)
    t["day"] = t.ts // 86400
    pc = t.groupby(["sender", "receiver"]).agg(n=("tx", "size"), d=("day", "nunique")).reset_index()
    cand = pc[(pc.n >= 3) & (pc.d >= 3)].reset_index(drop=True)
    cand["uid"] = np.arange(len(cand))
    sp = t.merge(cand[["sender", "receiver", "uid"]], on=["sender", "receiver"])[["tx", "ts", "sender", "receiver", "uid"]]
    sp.columns = ["stx", "sts", "A", "B", "uid"]
    ins = t[["tx", "ts", "sender", "receiver", "usd_in"]].merge(sp, left_on="receiver", right_on="A")
    ins = ins[(ins.ts < ins.sts) & (ins.ts >= ins.sts - WIN) & (ins.sender != ins.B)]
    outs = t[["tx", "ts", "sender", "receiver"]].merge(sp, left_on="sender", right_on="B")
    outs = outs[(outs.ts > outs.sts) & (outs.ts <= outs.sts + WIN) & (outs.receiver != outs.A)]
    # keep units where at least 2 spine payments have both inputs and outputs
    ok_in = ins.groupby("uid").stx.nunique(); ok_out = outs.groupby("uid").stx.nunique()
    both = pd.concat([ins[["uid", "stx"]], outs[["uid", "stx"]]]).drop_duplicates()
    good = (ins[["uid", "stx"]].drop_duplicates().merge(outs[["uid", "stx"]].drop_duplicates()).groupby("uid").size())
    good = good[good >= 2].index
    ins, outs, sp = ins[ins.uid.isin(good)], outs[outs.uid.isin(good)], sp[sp.uid.isin(good)]
    mem = pd.concat([sp[["uid", "stx"]].rename(columns={"stx": "tx"}), ins[["uid", "tx"]], outs[["uid", "tx"]]]).drop_duplicates()
    # wallet stats over the whole week
    L = pd.concat([t[["sender", "receiver"]].set_axis(["w", "cp"], axis=1), t[["receiver", "sender"]].set_axis(["w", "cp"], axis=1)])
    ncp = L.groupby("w").cp.nunique()
    wal = pd.concat([sp[["uid", "A"]].set_axis(["uid", "w"], axis=1), sp[["uid", "B"]].set_axis(["uid", "w"], axis=1),
                     ins[["uid", "sender"]].set_axis(["uid", "w"], axis=1), outs[["uid", "receiver"]].set_axis(["uid", "w"], axis=1)]).drop_duplicates()
    U = pd.DataFrame(index=good)
    U["ncp"] = wal.assign(n=wal.w.map(ncp)).groupby("uid").n.mean()
    nun = wal.groupby("w").uid.nunique()
    spw = pd.concat([sp[["uid", "A"]].set_axis(["uid", "w"], axis=1), sp[["uid", "B"]].set_axis(["uid", "w"], axis=1)]).drop_duplicates()
    U["reuse"] = spw.assign(n=spw.w.map(nun)).groupby("uid").n.mean()
    # originators' inflow in the 24h before paying A
    inb = {w: (g.ts.to_numpy(), np.cumsum(g.usd_out.to_numpy())) for w, g in t.groupby("receiver")}
    def back(w, ts):
        a = inb.get(w)
        if a is None:
            return 0.0
        T, C = a; i = np.searchsorted(T, ts, "left"); j = np.searchsorted(T, ts - BACK, "left")
        return (C[i - 1] if i > 0 else 0) - (C[j - 1] if j > 0 else 0)
    ins = ins.assign(b=[back(w, s) for w, s in zip(ins.sender, ins.ts)])
    g = ins.groupby("uid")
    U["back"] = g.b.sum() / g.usd_in.sum()
    U["score"] = ((-U.ncp).rank(pct=True) + (-U.reuse).rank(pct=True) + (-U.back).rank(pct=True)) / 3
    s = mem.assign(score=mem.uid.map(U.score)).groupby("tx").score.max()
    return t.tx.map(s).fillna(0).set_axis(t.tx).reindex(t.tx), U


def main():
    from sklearn.metrics import roc_auc_score, average_precision_score
    tr, _ = load_split("train")
    s_tr, U = unit_scores(tr)
    lab = load_labels().set_index("tx")
    oof = pd.DataFrame({"tx": tr.tx.to_numpy(), "score": s_tr.reindex(tr.tx).to_numpy()})
    oof.to_parquet("data/oof/hard_odd.parquet", index=False)
    d = oof.join(lab, on="tx")
    r = d[d.family.isin(["aggregate_relay", "distributed_relay"])]
    print("units", len(U), "relay rows covered", (r.score > 0).mean().round(3), "hard covered", (r[r.label == 1].score > 0).mean().round(3))
    print("within-relay AUC %.3f AP %.4f" % (roc_auc_score(r.label, r.score), average_precision_score(r.label, r.score)))
    rs = r.sort_values("score", ascending=False)
    print("hard in top 50/100/200:", [int(rs.label.head(k).sum()) for k in (50, 100, 200)])
    print("all-train AP %.4f" % average_precision_score(d.label, d.score))
    sc = r.groupby(["cohort_id", "scenario_id"]).agg(s=("score", "mean"), y=("label", "max")).reset_index()
    sc["rk"] = sc.groupby("cohort_id").s.rank(ascending=False)
    print("positive scenario rank in cohort:", sc[sc.y == 1].rk.tolist(), "rank1:", int((sc[sc.y == 1].rk == 1).sum()))
    te, _ = load_split("test")
    s_te, Ut = unit_scores(te)
    raw = s_te.reindex(te.tx).to_numpy()
    print("test units", len(Ut), "payments in units", int((raw > 0).sum()), "score>=0.9", int((raw >= 0.9).sum()), "score>=0.95", int((raw >= 0.95).sum()))
    try:
        tb = pd.read_csv("submissions/v17_realmotif.csv").set_index("TRANSACTION_HASH").score.reindex(te.tx).fillna(0).to_numpy()
    except Exception:
        tb = np.zeros(len(te))
    write("hard_odd", te, raw, tb)


if __name__ == "__main__":
    main()
