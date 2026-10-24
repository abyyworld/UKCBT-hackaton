"""Graph diffusion of v17 scores over each week's own payment graph (no new model training).

Two payments are linked when they share a wallet and happen within T seconds of each other
(only the k nearest payments of that wallet on each side, so busy wallets and relay hubs cannot
blow up a payment's degree; links through busy wallets are also down-weighted). Link weight
is exp(-gap / T). Optionally only payments on first-time counterparty pairs (pairs seen once in
the week) are linked. v17 scores, as percentile ranks, seed a personalised-PageRank style spread
s <- (1 - a) * seed + a * W s. T, a, the first-time restriction, the seed sharpness and the final
mix with v17 are picked on TRAIN (v17 OOF as seed; labels only to choose the setting), then the
same setting is applied to test (v17 test scores as seed).

    .venv/bin/python -W ignore scripts/diffuse_run.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from src.cv import report, training_frame  # noqa: E402
from src.load import load_split  # noqa: E402
from src.predict import write  # noqa: E402

K_NEAR = 5      # nearest payments per wallet on each side in time
N_ITER = 15


def pct(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def build_edges(txs):
    """All candidate links (i, j, gap, wallet size, both-first-time) up to the widest window."""
    n = len(txs)
    idx = np.arange(n)
    snd, rcv = txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    long = pd.DataFrame({"node": np.r_[idx, idx[snd != rcv]],
                         "wallet": np.r_[snd, rcv[snd != rcv]],
                         "ts": np.r_[txs["ts"].to_numpy(), txs["ts"].to_numpy()[snd != rcv]]})
    long["nw"] = long.groupby("wallet")["node"].transform("size")
    long = long.sort_values(["wallet", "ts"]).reset_index(drop=True)
    pair_n = txs.groupby(["sender", "receiver"])["tx"].transform("size").to_numpy()
    first = pair_n == 1
    w, t, nd, nw = long["wallet"].to_numpy(), long["ts"].to_numpy(), long["node"].to_numpy(), long["nw"].to_numpy()
    out = []
    for d in range(1, K_NEAR + 1):
        same = w[:-d] == w[d:]
        a, b = nd[:-d][same], nd[d:][same]
        gap = (t[d:] - t[:-d])[same]
        keep = (gap <= 7200) & (a != b)
        out.append(pd.DataFrame({"i": a[keep], "j": b[keep], "gap": gap[keep], "nw": nw[:-d][same][keep]}))
    e = pd.concat(out, ignore_index=True)
    e["ft"] = first[e["i"].to_numpy()] & first[e["j"].to_numpy()]
    return e, n


def make_W(e, n, T, ft_only):
    m = (e["gap"].to_numpy() <= T)
    if ft_only:
        m &= e["ft"].to_numpy()
    i, j = e["i"].to_numpy()[m], e["j"].to_numpy()[m]
    wt = np.exp(-e["gap"].to_numpy()[m] / T) / np.sqrt(e["nw"].to_numpy()[m])
    W = sp.coo_matrix((np.r_[wt, wt], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    iso = deg == 0
    W = W + sp.diags(iso.astype(float))  # isolated payments keep their own score
    deg = np.where(iso, 1.0, deg)
    return sp.diags(1.0 / deg) @ W


def diffuse(W, seed, a):
    s = seed.copy()
    for _ in range(N_ITER):
        s = (1 - a) * seed + a * (W @ s)
    return s


def score(e, n, v17, T, a, ft, p, mix):
    seed = pct(v17) ** p
    s = diffuse(make_W(e, n, T, ft), seed, a)
    return mix * pct(s) + (1 - mix) * pct(v17)


if __name__ == "__main__":
    _, lab, _ = training_frame(["local"])
    y = lab["label"].to_numpy()
    tr, _ = load_split("train")
    v17_oof = pd.read_parquet("data/oof/v17_realmotif.parquet").set_index("tx")["oof"]
    tr_v17 = v17_oof.loc[tr["tx"]].to_numpy()
    lab_tr = lab.loc[tr["tx"]]
    y_tr = lab_tr["label"].to_numpy()
    e_tr, n_tr = build_edges(tr)
    print(f"train: {n_tr} payments, {len(e_tr)} candidate links, {e_tr['ft'].mean():.2%} first-time")
    print(f"v17 alone AP x100 {100 * average_precision_score(y_tr, tr_v17):.2f}")

    grid = list(itertools.product([300, 1800, 7200], [0.3, 0.5, 0.7], [False, True], [1, 4]))
    mixes = [0.3, 0.5, 0.7, 1.0]
    res = []
    for T, a, ft, p in grid:
        seed = pct(tr_v17) ** p
        s = pct(diffuse(make_W(e_tr, n_tr, T, ft), seed, a))
        for mix in mixes:
            f = mix * s + (1 - mix) * pct(tr_v17)
            res.append((T, a, ft, p, mix, 100 * average_precision_score(y_tr, f)))
    res = pd.DataFrame(res, columns=["T", "a", "ft", "p", "mix", "ap"]).sort_values("ap", ascending=False)
    print(res.head(15).to_string(index=False))
    best = res.iloc[0]
    T, a, ft, p, mix = int(best["T"]), float(best["a"]), bool(best["ft"]), int(best["p"]), float(best["mix"])
    print(f"chosen: T={T} a={a} first_time_only={ft} seed_power={p} mix_with_v17={mix}")

    best_tr = score(e_tr, n_tr, tr_v17, T, a, ft, p, mix)
    oof = pd.Series(best_tr, index=tr["tx"].to_numpy()).loc[lab.index].to_numpy()
    v17_al = v17_oof.loc[lab.index].to_numpy()
    report(v17_al, lab, "v17 alone")
    report(oof, lab, "diffuse best")
    report(0.5 * pct(oof) + 0.5 * pct(v17_al), lab, "diffuse 50/50 rank avg with v17")
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": oof}).to_parquet("data/oof/diffuse.parquet")

    te, _ = load_split("test")
    te_v17 = pd.read_csv("submissions/v17_realmotif.csv").set_index("TRANSACTION_HASH")["score"].loc[te["tx"]].to_numpy()
    e_te, n_te = build_edges(te)
    print(f"test: {n_te} payments, {len(e_te)} candidate links")
    raw = score(e_te, n_te, te_v17, T, a, ft, p, mix)
    print(f"test corr with v17 (spearman): {pd.Series(raw).corr(pd.Series(te_v17), method='spearman'):.4f}")
    write("diffuse", te, raw, te_v17)
