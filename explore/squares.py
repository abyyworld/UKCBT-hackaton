"""Prototype: closed squares of first-time pairs (s->r, s->r2, s2->r, s2->r2 all new pairs within W)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.load import load_labels, load_split


def squares(txs, W=3 * 3600):
    ts = txs["ts"].to_numpy().astype(float)
    snd, rcv = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    order = np.argsort(ts, kind="stable")
    novel = np.zeros(len(txs), bool)
    novel[order] = ~pd.Series(list(zip(snd, rcv))).iloc[order].duplicated().to_numpy()
    nv = np.flatnonzero(novel)
    first = {(snd[i], rcv[i]): ts[i] for i in nv}
    outs, ins = {}, {}
    for i in nv:
        outs.setdefault(snd[i], []).append((ts[i], rcv[i]))
        ins.setdefault(rcv[i], []).append((ts[i], snd[i]))
    sq, partners = np.zeros(len(txs)), np.zeros(len(txs))
    for i in nv:
        s, r, t = snd[i], rcv[i], ts[i]
        r2s = {x for tt, x in outs[s] if abs(tt - t) <= W and x != r}
        s2s = {x for tt, x in ins[r] if abs(tt - t) <= W and x != s}
        c, ps = 0, set()
        for s2 in s2s:
            for r2 in r2s:
                t2 = first.get((s2, r2))
                if t2 is not None and abs(t2 - t) <= W:
                    c += 1
                    ps.add(s2)
        sq[i], partners[i] = c, len(ps)
    return pd.DataFrame({"sq": sq, "sq_partners": partners, "novel": novel.astype(float)}, index=txs["tx"])


if __name__ == "__main__":
    t, _ = load_split("train")
    lab = load_labels().set_index("tx")
    for W in (3600, 3 * 3600):
        f = squares(t, W).join(lab)
        g = f.difficulty.where(f.difficulty != "hard")
        print(W, pd.crosstab(f.sq.clip(0, 4), g))
        print(pd.crosstab(f.sq.clip(0, 4), f.family.where(f.difficulty == "medium")))
        m = f.difficulty.isin(["medium", "normal"])
        print("medium vs normal AUC", round(roc_auc_score(f.difficulty[m] == "medium", f.sq[m]), 3),
              "AP", round(average_precision_score(f.difficulty[m] == "medium", f.sq[m]), 3))
