"""Prototype: multi-hop timing chains and repeated paths around relay hubs."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.load import load_labels, load_split


def chain_feats(t, W=3 * 3600):
    t = t.sort_values("ts").reset_index(drop=True)
    ts = t["ts"].to_numpy().astype(float)
    out_by = {w: g.index.to_numpy() for w, g in t.groupby("sender")}
    in_by = {w: g.index.to_numpy() for w, g in t.groupby("receiver")}
    snd, rcv = t["sender"].to_numpy(), t["receiver"].to_numpy()

    def nxt(i):  # payments by my receiver within W after me
        c = out_by.get(rcv[i], np.array([], int))
        return c[(ts[c] > ts[i]) & (ts[c] <= ts[i] + W)]

    def prv(i):  # payments into my sender within W before me
        c = in_by.get(snd[i], np.array([], int))
        return c[(ts[c] < ts[i]) & (ts[c] >= ts[i] - W)]

    n = len(t)
    down1 = np.zeros(n); down2 = np.zeros(n); up1 = np.zeros(n); up2 = np.zeros(n)
    path_rep = np.zeros(n)
    # path triple counts: (s, r, r2) where r pays r2 within W after receiving from s
    triples = {}
    nx = [nxt(i) for i in range(n)]
    for i in range(n):
        for j in nx[i]:
            k = (snd[i], rcv[i], rcv[j])
            triples.setdefault(k, set()).add(int(ts[i] // 86400))
    for i in range(n):
        a = nx[i]
        down1[i] = len(a)
        down2[i] = sum(len(nx[j]) > 0 for j in a)
        p = prv(i)
        up1[i] = len(p)
        up2[i] = sum(len(prv(j)) > 0 for j in p)
        path_rep[i] = max([len(triples.get((snd[i], rcv[i], rcv[j]), ())) for j in a] or [0])
    f = pd.DataFrame({"down1": down1, "down2": down2, "up1": up1, "up2": up2, "path_days": path_rep}, index=t["tx"])
    f["depth"] = (f.down1 > 0).astype(int) + (f.down2 > 0) + (f.up1 > 0) + (f.up2 > 0)
    return f


if __name__ == "__main__":
    t, _ = load_split("train")
    lab = load_labels().set_index("tx")
    f = chain_feats(t).join(lab)
    r = f[f.family.isin(["aggregate_relay", "distributed_relay"])]
    for c in ["down1", "down2", "up1", "up2", "path_days", "depth"]:
        print(c, r.groupby("label")[c].mean().round(2).to_dict(), "AUC", round(roc_auc_score(r.label, r[c]), 3),
              "AP", round(average_precision_score(r.label, r[c]), 3))
