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

GAP = 3 * 3600
def build_units(t):
    """episode units: per hub, union senders that co-burst into it; attach hub out-payments to the last preceding burst's unit."""
    t = t.sort_values("ts")
    rc = t.groupby("receiver").agg(ns=("sender", "nunique"), d=("ts", lambda x: (x // 86400).nunique()))
    outc = t.groupby("sender").size()
    hubs = rc[(rc.ns >= 3) & (rc.d >= 3)].index.intersection(outc[outc >= 2].index)
    mem = []
    for h in hubs:
        i = t[t.receiver == h]; o = t[t.sender == h]
        ts = i.ts.to_numpy(); b = np.concatenate([[0], np.cumsum(np.diff(ts) > GAP)])
        par = {}
        def f(x):
            while par.get(x, x) != x: x = par[x]
            return x
        for bid, g in i.groupby(b):
            s = list(g.sender.unique())
            if len(s) < 2: continue
            r0 = f(s[0])
            for x in s[1:]:
                rx = f(x)
                if rx != r0: par[rx] = r0
        comp = {s: f(s) for s in i.sender.unique()}
        csz = pd.Series(comp).value_counts()
        ii = i.assign(c=i.sender.map(comp)); ii = ii[ii.c.map(csz) >= 2]
        if ii.empty: continue
        last = ii[["ts", "c"]].sort_values("ts")
        oo = o.sort_values("ts")
        oo = pd.merge_asof(oo, last.rename(columns={"ts": "lts"}), left_on="ts", right_on="lts", direction="backward")
        oo = oo[(oo.ts - oo.lts) < 2 * 86400].dropna(subset=["c"])
        u = pd.concat([ii[["tx", "c"]], oo[["tx", "c"]]]); u["unit"] = h + "|" + u.c.astype(str); u["hub"] = h
        mem.append(u[["tx", "unit", "hub"]])
    return pd.concat(mem)


def outlier(U, coh):
    X = np.log1p(U.clip(lower=0)).fillna(0)
    z = X.groupby(coh).transform(lambda c: (c - c.median()) / np.maximum(1.4826 * (c - c.median()).abs().median(), c.std() + 1e-6)).abs()
    return pd.Series(np.sort(z.clip(upper=50).to_numpy(), axis=1)[:, -3:].mean(axis=1), index=z.index)

def run_units(t):
    mem = build_units(t)
    tt = t.set_index("tx")
    rows = {}
    for uid, g in mem.groupby("unit"):
        rows[uid] = stats(tt.loc[g.tx.unique()].reset_index(), g.hub.iloc[0])
    U = pd.DataFrame(rows).T.astype(float)
    U = U[U.n >= 8]
    Xs = np.log1p(U.clip(lower=0)).fillna(0); Xs = (Xs - Xs.mean()) / (Xs.std() + 1e-9)
    coh = pd.Series(KMeans(12, n_init=5, random_state=0).fit_predict(Xs), index=U.index)
    return mem[mem.unit.isin(U.index)], U, coh, outlier(U, coh)

if __name__ == "__main__":
    tx, _ = load_split("train"); lab = load_labels()
    mem, U, coh, s = run_units(tx)
    m = mem.merge(lab, on="tx")
    lr = m.groupby("unit").apply(lambda g: (g.label == 1).mean())
    rk = s.groupby(coh).rank(ascending=False); sz = coh.map(coh.value_counts())
    print("TRAIN units", len(U), "cohort sizes", coh.value_counts().tolist())
    L = lr[lr > 0.5].index.intersection(U.index)
    print("TRAIN laundering-dominated units", len(L), "rank pct within cohort", (rk[L] / sz[L]).round(2).tolist())
    win = s.groupby(coh).idxmax(); print("TRAIN cohort top units laundering share", lr.reindex(win.values).round(2).tolist())
    te, _ = load_split("test")
    mem, U, coh, s = run_units(te)
    print("TEST units", len(U), "payments/unit q", U.n.quantile([.25, .5, .75]).tolist(), "cohort sizes", coh.value_counts().tolist())
    best = s.groupby(coh).apply(lambda c: c.nlargest(2).to_numpy())
    print("TEST top1, top2 per cohort", [(round(a[0], 1), round(a[1], 1)) for a in best])
    win = s.groupby(coh).idxmax()
    flag = mem[mem.unit.isin(win.values)].assign(score=lambda d: d.unit.map(s)).groupby("tx").score.max()
    out = pd.DataFrame({"TRANSACTION_HASH": te.tx}); out["score"] = out.TRANSACTION_HASH.map(flag).fillna(0)
    out.to_csv("submissions/clue_odd.csv", index=False)
    print("rows", len(out), "flagged", int((out.score > 0).sum()))
