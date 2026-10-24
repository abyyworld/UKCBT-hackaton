"""Temporal edge-classification GNN: wallets are nodes, payments are edges.

    python scripts/gnn_model.py

Edge-aware message passing (GINE style) with separate weights for the incoming and
outgoing direction, then an edge classifier on [h_src, h_dst, e]. Out-of-fold AP on
train (5 folds grouped by scenario), then fit on all of train and score the test week.
Writes data/oof/gnn.parquet and submissions/gnn.csv. No amount-evenness features.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.cv import report
from src.load import load_labels, load_split
from src.predict import write

SEED = 0
HIDDEN = 64
LAYERS = 3
MAX_EPOCHS = int(os.environ.get("GNN_EPOCHS", 300))
EVAL_EVERY = 10
PATIENCE = 6  # evaluations without improvement
N_SEEDS = int(os.environ.get("GNN_SEEDS", 2))
WINDOWS = (120, 300, 900, 3600)

torch.set_num_threads(4)


def tempo_counts(wallet_codes_a, wallet_codes_b, ts, w):
    """For each payment, number of other payments touching its sender (and receiver) within +-w seconds."""
    n = len(ts)
    inv_w = np.concatenate([wallet_codes_a, wallet_codes_b])
    inv_t = np.concatenate([ts, ts])
    order = np.lexsort((inv_t, inv_w))
    sw, st = inv_w[order], inv_t[order]
    key = sw.astype(np.float64) * 1e7 + (st - ts.min())
    out = []
    for codes in (wallet_codes_a, wallet_codes_b):
        q = codes.astype(np.float64) * 1e7 + (ts - ts.min())
        hi = np.searchsorted(key, q + w, side="right")
        lo = np.searchsorted(key, q - w, side="left")
        out.append(hi - lo - 1)  # minus the payment itself
    return out[0], out[1]


def build_graph(txs):
    wallets = pd.Index(pd.unique(np.concatenate([txs["sender"].to_numpy(), txs["receiver"].to_numpy()])))
    s = wallets.get_indexer(txs["sender"]).astype(np.int64)
    d = wallets.get_indexer(txs["receiver"]).astype(np.int64)
    ts = txs["ts"].to_numpy().astype(np.int64)
    n_nodes, n = len(wallets), len(txs)

    usd = txs["usd_in"].fillna(txs["usd_out"]).fillna(0).to_numpy()
    tod = (ts % 86400) / 86400 * 2 * np.pi
    pair = pd.Series(s * n_nodes + d)
    first_pair = (~pd.DataFrame({"p": pair, "t": ts}).sort_values("t", kind="stable")["p"].duplicated()).sort_index()
    pair_count = pair.map(pair.value_counts()).to_numpy()
    rev = pd.Series(d * n_nodes + s)
    has_reverse = rev.isin(set(pair)).to_numpy()

    feats = {
        "log_usd": np.log1p(usd),
        "tod_sin": np.sin(tod), "tod_cos": np.cos(tod),
        "n_legs": np.log1p(txs["n_legs"].to_numpy()),
        "is_conv": txs["is_conversion"].to_numpy().astype(float),
        "first_pair": first_pair.to_numpy().astype(float),
        "pair_count": np.log1p(pair_count),
        "has_reverse": has_reverse.astype(float),
        "success": txs["success"].to_numpy().astype(float),
        "has_tag": txs["dest_tag"].notna().to_numpy().astype(float),
        "day_frac": (ts - ts.min()) / max(1, ts.max() - ts.min()),
    }
    for w in WINDOWS:
        a, b = tempo_counts(s, d, ts, w)
        feats[f"tempo_s_{w}"] = np.log1p(a)
        feats[f"tempo_d_{w}"] = np.log1p(b)
    # time gap to the previous / next payment on the same sender and receiver
    for name, codes in (("s", s), ("d", d)):
        df = pd.DataFrame({"c": codes, "t": ts}).sort_values(["c", "t"], kind="stable")
        gp = df.groupby("c")["t"]
        prev = (df["t"] - gp.shift(1)).reindex(range(n))
        nxt = (gp.shift(-1) - df["t"]).reindex(range(n))
        feats[f"gap_prev_{name}"] = np.log1p(prev.fillna(7 * 86400).to_numpy())
        feats[f"gap_next_{name}"] = np.log1p(nxt.fillna(7 * 86400).to_numpy())
    E = np.column_stack(list(feats.values())).astype(np.float32)

    deg_out = np.bincount(s, minlength=n_nodes)
    deg_in = np.bincount(d, minlength=n_nodes)
    uniq = pd.DataFrame({"s": s, "d": d}).drop_duplicates()
    cp_out = np.bincount(uniq["s"], minlength=n_nodes)
    cp_in = np.bincount(uniq["d"], minlength=n_nodes)
    fp = feats["first_pair"]
    fp_share = (np.bincount(s, fp, n_nodes) + np.bincount(d, fp, n_nodes)) / np.maximum(1, deg_out + deg_in)
    tmin = np.full(n_nodes, np.inf)
    tmax = np.full(n_nodes, -np.inf)
    for c in (s, d):
        np.minimum.at(tmin, c, ts)
        np.maximum.at(tmax, c, ts)
    span = np.log1p(tmax - tmin)
    ratio = (deg_in + 1) / (deg_out + 1)
    X = np.column_stack([np.log1p(deg_out), np.log1p(deg_in), np.log1p(cp_out), np.log1p(cp_in),
                         fp_share, span, np.log(ratio), (deg_out == 0), (deg_in == 0)]).astype(np.float32)

    # standardise within the week, so the train and test weeks are each centred on themselves
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return {"x": torch.tensor(X), "e": torch.tensor(E), "src": torch.tensor(s), "dst": torch.tensor(d),
            "n_nodes": n_nodes, "tie": feats["log_usd"]}


def mlp(i, o, h=None):
    h = h or o
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class Layer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.msg_in = mlp(h, h)   # messages along the payment direction (sender -> receiver)
        self.msg_out = mlp(h, h)  # reverse direction (receiver -> sender)
        self.upd = mlp(3 * h, h)
        self.edge_upd = mlp(3 * h, h)
        self.norm_n = nn.LayerNorm(h)
        self.norm_e = nn.LayerNorm(h)

    def forward(self, h, e, src, dst, inv_in, inv_out):
        m_in = torch.zeros_like(h).index_add_(0, dst, self.msg_in(h[src] + e)) * inv_in
        m_out = torch.zeros_like(h).index_add_(0, src, self.msg_out(h[dst] + e)) * inv_out
        h = self.norm_n(h + self.upd(torch.cat([h, m_in, m_out], 1)))
        e = self.norm_e(e + self.edge_upd(torch.cat([h[src], h[dst], e], 1)))
        return h, e


class EdgeGNN(nn.Module):
    def __init__(self, nx, ne, h=HIDDEN, layers=LAYERS):
        super().__init__()
        self.node_in = mlp(nx, h)
        self.edge_in = mlp(ne, h)
        self.layers = nn.ModuleList([Layer(h) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, g):
        src, dst = g["src"], g["dst"]
        if "inv_in" not in g:
            n = g["n_nodes"]
            g["inv_in"] = (1 / torch.bincount(dst, minlength=n).clamp(min=1).float()).unsqueeze(1)
            g["inv_out"] = (1 / torch.bincount(src, minlength=n).clamp(min=1).float()).unsqueeze(1)
        h, e = self.node_in(g["x"]), self.edge_in(g["e"])
        for layer in self.layers:
            h, e = layer(h, e, src, dst, g["inv_in"], g["inv_out"])
        return self.head(torch.cat([h[src], h[dst], e], 1)).squeeze(1)


def train(g, y, fit_idx, seed, hold_idx=None, epochs=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EdgeGNN(g["x"].shape[1], g["e"].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    yt = torch.tensor(y, dtype=torch.float32)
    fit_t = torch.tensor(fit_idx)
    pos = y[fit_idx].mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(10.0, (1 - pos) / pos)))
    best, best_ep, best_state, bad = -1, 0, None, 0
    for ep in range(1, (epochs or MAX_EPOCHS) + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(g)[fit_t], yt[fit_t])
        loss.backward()
        opt.step()
        if hold_idx is not None and ep % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                p = model(g).numpy()[hold_idx]
            ap = average_precision_score(y[hold_idx], p)
            if ap > best:
                best, best_ep, bad = ap, ep, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(g)).numpy()
    return out, best_ep, best


def main():
    t0 = time.time()
    txs, _ = load_split("train")
    lab = load_labels().set_index("tx").loc[txs["tx"]]
    y = lab["label"].to_numpy().astype(np.float32)
    groups = lab["scenario_id"].fillna(pd.Series(lab.index, index=lab.index)).to_numpy()
    g = build_graph(txs)
    print(f"train graph: {g['n_nodes']} wallets, {len(txs)} payments, {g['e'].shape[1]} edge feats "
          f"({time.time() - t0:.0f}s)", flush=True)

    oof = np.zeros(len(y))
    epochs = []
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (fit, hold) in enumerate(folds.split(txs, y, groups)):
        p, ep, ap = train(g, y, fit, SEED + k, hold_idx=hold)
        oof[hold] = p[hold]
        epochs.append(ep)
        print(f"fold {k}: best epoch {ep}, hold AP {100 * ap:.2f} ({time.time() - t0:.0f}s)", flush=True)
    lab = lab.reset_index()
    report(oof, lab, "gnn OOF")
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": txs["tx"].to_numpy(), "oof": oof}).to_parquet("data/oof/gnn.parquet", index=False)

    test_txs, _ = load_split("test")
    gt = build_graph(test_txs)
    n_ep = int(np.mean(epochs))
    all_idx = np.arange(len(y))
    raw = np.zeros(len(test_txs))
    for s in range(N_SEEDS):
        model_seed = 100 + s
        torch.manual_seed(model_seed)
        # fit on all of train, then score the test graph with the same weights
        model = EdgeGNN(g["x"].shape[1], g["e"].shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
        yt = torch.tensor(y)
        pos = y.mean()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(10.0, (1 - pos) / pos)))
        for _ in range(n_ep):
            model.train()
            opt.zero_grad()
            loss_fn(model(g)[all_idx], yt[all_idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            raw += pd.Series(model(gt).numpy()).rank().to_numpy() / len(test_txs)
        print(f"final seed {s}: {n_ep} epochs ({time.time() - t0:.0f}s)", flush=True)
    write("gnn", test_txs, raw / N_SEEDS, pd.Series(gt["tie"]).rank().to_numpy())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
