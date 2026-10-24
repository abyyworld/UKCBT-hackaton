"""Transductive odd-one-out over relay units. Train check uses oracle scenario units; test uses hub units."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from src.load import load_split, load_labels

def stats(g, hub=None):
    g = g.sort_values("ts"); d = g.ts // 86400
    if hub is None:
        hub = pd.concat([g.sender, g.receiver]).mode().iloc[0]
    i, o = g[g.receiver == hub], g[g.sender == hub]
    gap = np.diff(g.ts.to_numpy()) if len(g) > 1 else np.array([0])
    r = dict(n=len(g), nin=len(i), nout=len(o), ns=g.sender.nunique(), nr=g.receiver.nunique(), days=d.nunique(),
             span=g.ts.max() - g.ts.min(), gmed=np.median(gap), gmax=gap.max(), usd_med=g.usd_in.median(),
             usd_sum=g.usd_in.sum(), usd_cv=g.usd_in.std() / (g.usd_in.mean() + 1e-9),
             in_sum=i.usd_in.sum(), out_sum=o.usd_out.sum(), ratio=(o.usd_out.sum() + 1) / (i.usd_in.sum() + 1),
             conv=g.is_conversion.mean(), legs=g.n_legs.mean(), fee=g.fee_xrp.mean(), nai=g.asset_in.nunique(),
             nao=g.asset_out.nunique(), xrp=(g.asset_in == "XRP").mean(),
             delay=(o.ts.median() - i.ts.median()) if len(i) and len(o) else 0,
             churn=i.groupby(i.ts // 86400).sender.apply(set).pipe(lambda s: np.mean([len(a ^ b) / max(len(a | b), 1) for a, b in zip(s[:-1], s[1:])]) if len(s) > 1 else 0),
             hour=(g.ts % 86400).std(), ins_per_day=len(i) / max(i.ts.floordiv(86400).nunique(), 1))
    return r

def outlier(U, coh):
    X = np.log1p(U.clip(lower=0)).fillna(0)
    z = X.groupby(coh).transform(lambda c: (c - c.median()) / (1.4826 * (c - c.median()).abs().median() + 1e-6)).abs()
    return z.clip(upper=50).max(axis=1)

if __name__ == "__main__":
    tx, _ = load_split("train"); lab = load_labels(); t = tx.merge(lab, on="tx")
    rel = t[t.family.isin(["aggregate_relay", "distributed_relay"])]
    U = pd.DataFrame({s: stats(g) for s, g in rel.groupby("scenario_id")}).T
    meta = rel.groupby("scenario_id").agg(y=("label", "max"), coh=("cohort_id", "first"))
    sc = outlier(U, meta.coh)
    meta["sc"] = sc; meta["rk"] = meta.groupby("coh").sc.rank(ascending=False)
    top = meta.groupby("coh").sc.apply(lambda s: s.nlargest(2).to_numpy())
    print("TRAIN oracle units: laundering rank within cohort:", meta[meta.y == 1].rk.tolist(), "cohort size", meta.groupby("coh").size().tolist())
    print("top1/top2 sep per cohort:", [round(a[0] / a[1], 2) for a in top])
    # test: hub units
    te, _ = load_split("test")
    te["day"] = te.ts // 86400
    rc = te.groupby("receiver").agg(ns=("sender", "nunique"), d=("day", "nunique"), n=("tx", "size"))
    outc = te.groupby("sender").size()
    hubs = rc[(rc.ns >= 3) & (rc.d >= 3)].index.intersection(outc[outc >= 3].index)
    rows, mem = {}, []
    for h in hubs:
        g = te[(te.receiver == h) | (te.sender == h)]
        if len(g) > 200: continue
        rows[h] = stats(g, h); mem.append(pd.DataFrame({"tx": g.tx, "h": h}))
    UT = pd.DataFrame(rows).T.astype(float); mem = pd.concat(mem)
    print("TEST hub units:", len(UT), "size quantiles", UT.n.quantile([.1, .5, .9]).tolist())
    Xs = np.log1p(UT.clip(lower=0)).fillna(0); Xs = (Xs - Xs.mean()) / (Xs.std() + 1e-9)
    coh = pd.Series(KMeans(12, n_init=5, random_state=0).fit_predict(Xs), index=UT.index)
    s = outlier(UT, coh)
    best = s.groupby(coh).apply(lambda c: c.nlargest(2).to_numpy())
    print("TEST cohort sizes", coh.value_counts().tolist())
    print("TEST top1, top2 per cohort", [(round(a[0], 1), round(a[1], 1)) for a in best])
    win = s.groupby(coh).idxmax()
    flag = mem[mem.h.isin(win.values)].assign(score=lambda d: d.h.map(s)).groupby("tx").score.max()
    out = pd.DataFrame({"TRANSACTION_HASH": te.tx}); out["score"] = out.TRANSACTION_HASH.map(flag).fillna(0)
    out.to_csv("submissions/clue_odd.csv", index=False)
    print("rows", len(out), "flagged", int((out.score > 0).sum()))
