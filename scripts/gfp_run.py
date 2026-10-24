"""GFP baseline (tag gfp): a from-scratch take on IBM's Graph Feature Preprocessor + gradient boosting.

For every payment we look at what the sender and the receiver did in time windows before and after it
(counts of outgoing and incoming payments, how many were to brand-new counterparties, amount sums and
spreads, time to the nearest neighbouring payment), count two-hop scatter-gather paths and shared
sources/sinks of the pair, count short time-respecting cycles through the payment, and add the raw
payment fields. One LightGBM is trained on top. Features come from raw txs only (no src/features) and
are cached to data/cache/gfp_<split>.parquet.

    .venv/bin/python -W ignore scripts/gfp_run.py
"""
import os
import sys
import time
import zlib
from bisect import bisect_left, bisect_right
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.getcwd())
from src.cv import report, training_frame  # noqa: E402
from src.load import load_split  # noqa: E402
from src.predict import write  # noqa: E402

WINDOWS = [120, 600, 1800, 3600, 7200, 21600, 86400]
SG_WINDOWS = [3600, 7200, 21600]
CYCLE_W = 21600
CYCLE_CAP = 4000
BIG = 10**8


def window_stats(F, role_codes, t, ev_wallet, ev_t, ev_lu, ev_new, tag, own_lu, links):
    order = np.lexsort((ev_t, ev_wallet))
    key = ev_wallet[order].astype(np.int64) * BIG + ev_t[order]
    lu = ev_lu[order]
    c_lu = np.concatenate([[0.0], np.cumsum(lu)])
    c_lu2 = np.concatenate([[0.0], np.cumsum(lu * lu)])
    c_new = np.concatenate([[0], np.cumsum(ev_new[order])])
    base = role_codes.astype(np.int64) * BIG
    q_lt = np.searchsorted(key, base + t, "left")
    q_le = np.searchsorted(key, base + t, "right")
    F[f"{tag}_same_t"] = q_le - q_lt
    for w in WINDOWS:
        for side, lo, hi in (("b", np.searchsorted(key, base + t - w, "left"), q_lt),
                             ("a", q_le, np.searchsorted(key, base + t + w, "right"))):
            n = hi - lo
            s1 = c_lu[hi] - c_lu[lo]
            s2 = c_lu2[hi] - c_lu2[lo]
            mean = np.where(n > 0, s1 / np.maximum(n, 1), np.nan)
            var = np.where(n > 1, s2 / np.maximum(n, 1) - mean ** 2, np.nan)
            p = f"{tag}_{side}{w}"
            F[p + "_n"] = n
            F[p + "_new"] = c_new[hi] - c_new[lo]
            F[p + "_mean"] = mean
            F[p + "_std"] = np.sqrt(np.clip(var, 0, None))
    # gap to nearest previous / next event of this wallet (excluding same-second events)
    prev_i = q_lt - 1
    ok = (prev_i >= 0) & (key[np.clip(prev_i, 0, None)] // BIG == role_codes)
    F[f"{tag}_gap_prev"] = np.where(ok, t - key[np.clip(prev_i, 0, None)] % BIG, np.nan)
    nxt = np.clip(q_le, 0, len(key) - 1)
    ok = (q_le < len(key)) & (key[nxt] // BIG == role_codes)
    F[f"{tag}_gap_next"] = np.where(ok, key[nxt] % BIG - t, np.nan)
    # amount of the nearest previous / next event (does money move on at a similar size?)
    F[f"{tag}_dlu_prev"] = np.where(F[f"{tag}_gap_prev"].notna(), lu[np.clip(prev_i, 0, None)] - own_lu, np.nan)
    F[f"{tag}_dlu_next"] = np.where(F[f"{tag}_gap_next"].notna(), lu[nxt] - own_lu, np.nan)
    # gaps to the nearest first-time-pair event of this wallet (tempo of new counterparties)
    m = ev_new[order] > 0
    kn = key[m]
    a = np.searchsorted(kn, base + t, "left") - 1
    ok = (a >= 0) & (kn[np.clip(a, 0, None)] // BIG == role_codes)
    F[f"{tag}_gap_prev_new"] = np.where(ok, t - kn[np.clip(a, 0, None)] % BIG, np.nan)
    b = np.searchsorted(kn, base + t, "right")
    bb = np.clip(b, 0, len(kn) - 1)
    ok = (b < len(kn)) & (kn[bb] // BIG == role_codes)
    F[f"{tag}_gap_next_new"] = np.where(ok, kn[bb] % BIG - t, np.nan)
    # gaps to the 2nd and 3rd nearest first-time-pair events (is there a burst of new counterparties?)
    for k in (1, 2):
        ak, bk = a - k, b + k
        ok = (ak >= 0) & (kn[np.clip(ak, 0, None)] // BIG == role_codes)
        F[f"{tag}_gap_prev_new{k + 1}"] = np.where(ok, t - kn[np.clip(ak, 0, None)] % BIG, np.nan)
        bc = np.clip(bk, 0, len(kn) - 1)
        ok = (bk < len(kn)) & (kn[bc] // BIG == role_codes)
        F[f"{tag}_gap_next_new{k + 1}"] = np.where(ok, kn[bc] % BIG - t, np.nan)
    # which payment is the neighbouring one (used for one round of neighbour feature passing)
    on = order[m]
    links[f"{tag}_prev"] = np.where(F[f"{tag}_gap_prev"].notna(), order[np.clip(prev_i, 0, None)], -1)
    links[f"{tag}_next"] = np.where(F[f"{tag}_gap_next"].notna(), order[nxt], -1)
    links[f"{tag}_prev_new"] = np.where(F[f"{tag}_gap_prev_new"].notna(), on[np.clip(a, 0, None)], -1)
    links[f"{tag}_next_new"] = np.where(F[f"{tag}_gap_next_new"].notna(), on[bb], -1)


def graph_loops(s, r, t):
    """Scatter-gather, shared source/sink and time-respecting cycle counts (bounded python loops)."""
    n = len(s)
    OT, OD, IT, ID = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    order = np.argsort(t, kind="stable")
    for i in order:
        OT[s[i]].append(t[i]); OD[s[i]].append(r[i])
        IT[r[i]].append(t[i]); ID[r[i]].append(s[i])
    out = {}
    for w in SG_WINDOWS:
        out[f"sg_mid{w}"] = np.zeros(n); out[f"sg_mid_ord{w}"] = np.zeros(n)
        out[f"co_src{w}"] = np.zeros(n); out[f"co_sink{w}"] = np.zeros(n)
    for L in range(2, 6):
        out[f"cyc_f{L}"] = np.zeros(n); out[f"cyc_b{L}"] = np.zeros(n)
    out["cyc_f_1h"] = np.zeros(n); out["cyc_b_1h"] = np.zeros(n)
    out["cyc_f_2h"] = np.zeros(n); out["cyc_b_2h"] = np.zeros(n)
    out["cyc_min_len"] = np.full(n, np.nan); out["cyc_capped"] = np.zeros(n)

    for i in range(n):
        si, ri, ti = s[i], r[i], t[i]
        for w in SG_WINDOWS:
            lo, hi = ti - w, ti + w
            # sender's out-neighbours m (in window) that also paid the receiver (in window)
            ot, od = OT[si], OD[si]
            a, b = bisect_left(ot, lo), bisect_right(ot, hi)
            mids, mids_ord = set(), set()
            for k in range(a, b):
                m = od[k]
                if m == ri or m == si or m in mids:
                    continue
                it, idd = IT[ri], ID[ri]
                c, d = bisect_left(it, lo), bisect_right(it, hi)
                for q in range(c, d):
                    if idd[q] == m:
                        mids.add(m)
                        if it[q] >= ot[k]:
                            mids_ord.add(m)
            out[f"sg_mid{w}"][i] = len(mids); out[f"sg_mid_ord{w}"][i] = len(mids_ord)
            # common sources x -> s and x -> r
            it_s, id_s = IT[si], ID[si]
            src_s = set(id_s[bisect_left(it_s, lo):bisect_right(it_s, hi)])
            it_r, id_r = IT[ri], ID[ri]
            src_r = set(id_r[bisect_left(it_r, lo):bisect_right(it_r, hi)])
            src_r.discard(si)
            out[f"co_src{w}"][i] = len(src_s & src_r)
            # common sinks s -> x and r -> x
            snk_s = set(od[a:b]); snk_s.discard(ri)
            ot_r, od_r = OT[ri], OD[ri]
            snk_r = set(od_r[bisect_left(ot_r, lo):bisect_right(ot_r, hi)])
            out[f"co_sink{w}"][i] = len(snk_s & snk_r)
        # forward cycles: payment opens it, r -> ... -> s with increasing times in (t, t + W]
        budget = CYCLE_CAP
        minlen = 99
        tend = ti + CYCLE_W
        stack = [(ri, ti, 1)]
        while stack and budget > 0:
            u, lt, L = stack.pop()
            ot, od = OT[u], OD[u]
            for k in range(bisect_right(ot, lt), bisect_right(ot, tend)):
                budget -= 1
                v = od[k]
                if v == si:
                    out[f"cyc_f{L + 1}"][i] += 1
                    minlen = min(minlen, L + 1)
                    if ot[k] - ti <= 3600:
                        out["cyc_f_1h"][i] += 1
                    if ot[k] - ti <= 7200:
                        out["cyc_f_2h"][i] += 1
                elif L + 1 < 5 and v != ri:
                    stack.append((v, ot[k], L + 1))
        # backward cycles: payment closes it, r <- ... <- s with decreasing times in [t - W, t)
        tstart = ti - CYCLE_W
        stack = [(si, ti, 1)]
        while stack and budget > 0:
            u, lt, L = stack.pop()
            it, idd = IT[u], ID[u]
            for k in range(bisect_left(it, tstart), bisect_left(it, lt)):
                budget -= 1
                v = idd[k]
                if v == ri:
                    out[f"cyc_b{L + 1}"][i] += 1
                    minlen = min(minlen, L + 1)
                    if ti - it[k] <= 3600:
                        out["cyc_b_1h"][i] += 1
                    if ti - it[k] <= 7200:
                        out["cyc_b_2h"][i] += 1
                elif L + 1 < 5 and v != si:
                    stack.append((v, it[k], L + 1))
        if minlen < 99:
            out["cyc_min_len"][i] = minlen
        out["cyc_capped"][i] = float(budget <= 0)
    return out


def features(split):
    path = f"data/cache/gfp_{split}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    t0 = time.time()
    txs, _ = load_split(split)
    txs = txs.reset_index(drop=True)
    t = txs["ts"].to_numpy(np.int64)
    t = t - t.min()
    wallets = pd.unique(np.concatenate([txs["sender"].to_numpy(), txs["receiver"].to_numpy()]))
    code = {w: i for i, w in enumerate(wallets)}
    s = txs["sender"].map(code).to_numpy(np.int64)
    r = txs["receiver"].map(code).to_numpy(np.int64)
    usd = txs["usd_out"].fillna(0).to_numpy(float)
    lu = np.log1p(np.clip(usd, 0, None))
    n = len(txs)

    F = pd.DataFrame({"tx": txs["tx"].to_numpy()})
    # raw edge fields
    F["usd_out"] = usd
    F["usd_in"] = txs["usd_in"].to_numpy(float)
    F["lu"] = lu
    F["usd_ratio"] = txs["usd_out"] / txs["usd_in"].replace(0, np.nan)
    F["amount_out"] = txs["amount_out"].to_numpy(float)
    F["amt_round"] = (np.abs(F["amount_out"] - np.round(F["amount_out"])) < 1e-9).astype(int)
    F["usd_round10"] = (np.round(usd) % 10 == 0).astype(int)
    F["n_legs"] = txs["n_legs"].to_numpy()
    F["is_conv"] = txs["is_conversion"].astype(int).to_numpy()
    F["fee_xrp"] = txs["fee_xrp"].to_numpy(float)
    F["hour"] = (txs["ts"] % 86400) // 3600
    F["dow"] = (txs["ts"] // 86400) % 7
    cur = lambda a: a.map(lambda x: "XRP" if x == "XRP" else x.split(":")[-1])  # noqa: E731
    F["asset_out_h"] = cur(txs["asset_out"]).map(lambda x: zlib.crc32(x.encode()) % 997)
    F["asset_in_h"] = cur(txs["asset_in"]).map(lambda x: zlib.crc32(x.encode()) % 997)
    F["xrp_out"] = (txs["asset_out"] == "XRP").astype(int)

    # pair history
    order = np.lexsort((np.arange(n), t))
    pair = s * 1_000_000 + r
    ps = pd.Series(pair[order])
    first = np.zeros(n, bool)
    first[order] = ~ps.duplicated().to_numpy()
    prior = np.zeros(n)
    prior[order] = ps.groupby(ps).cumcount().to_numpy()
    F["pair_first"] = first.astype(int)
    F["pair_prior_n"] = prior
    total = pd.Series(pair).map(pd.Series(pair).value_counts()).to_numpy()
    F["pair_total_n"] = total
    pmin = pd.Series(t).groupby(pair).min()
    rev = r * 1_000_000 + s
    rev_first = pd.Series(rev).map(pmin).to_numpy(float)
    F["rev_before"] = (rev_first < t).astype(int)
    F["rev_gap"] = np.where(np.isnan(rev_first), np.nan, t - rev_first)
    ts_pair = pd.Series(t[order]).groupby(pair[order]).diff().to_numpy()
    gp = np.full(n, np.nan)
    gp[order] = ts_pair
    F["pair_gap_prev"] = gp
    # wallet totals over the whole split
    out_deg = pd.Series(s).value_counts()
    in_deg = pd.Series(r).value_counts()
    F["s_out_total"] = pd.Series(s).map(out_deg).fillna(0).to_numpy()
    F["s_in_total"] = pd.Series(s).map(in_deg).fillna(0).to_numpy()
    F["r_out_total"] = pd.Series(r).map(out_deg).fillna(0).to_numpy()
    F["r_in_total"] = pd.Series(r).map(in_deg).fillna(0).to_numpy()
    F["s_nuniq_out"] = pd.Series(s).map(pd.Series(r).groupby(s).nunique()).to_numpy()
    F["r_nuniq_in"] = pd.Series(r).map(pd.Series(s).groupby(r).nunique()).to_numpy()

    # GFP-style windowed vertex statistics
    newf = first.astype(float)
    links = {}
    for role, codes in (("s", s), ("r", r)):
        window_stats(F, codes, t, s, t, lu, newf, f"{role}out", lu, links)
        window_stats(F, codes, t, r, t, lu, newf, f"{role}in", lu, links)
    # windowed counts relative to the wallet's usual hourly rate over the split
    span_h = max(t.max() / 3600.0, 1.0)
    for tag, tot in (("sout", "s_out_total"), ("sin", "s_in_total"), ("rout", "r_out_total"), ("rin", "r_in_total")):
        rate = F[tot].to_numpy(float) / span_h
        for w in (600, 3600, 7200):
            F[f"{tag}_rel{w}"] = (F[f"{tag}_b{w}_n"] + F[f"{tag}_a{w}_n"]) / (rate * 2 * w / 3600 + 1e-3)
    # relative amount vs sender's usual
    F["lu_vs_sout_mean86400"] = lu - F["sout_b86400_mean"]
    print(f"[{split}] window features {F.shape[1]} cols, {time.time() - t0:.0f}s", flush=True)

    loops = graph_loops(s.tolist(), r.tolist(), t.tolist())
    for k, v in loops.items():
        F[k] = v
    F["cyc_f_all"] = sum(F[f"cyc_f{L}"] for L in range(2, 6))
    F["cyc_b_all"] = sum(F[f"cyc_b{L}"] for L in range(2, 6))
    # one round of neighbour passing: copy key features of the payment that moved the money on
    # (receiver's next outgoing) and of the one that brought it in (sender's previous incoming)
    pass_cols = ["pair_first", "pair_total_n", "lu", "rout_gap_next_new", "sin_gap_prev_new",
                 "rin_gap_prev_new", "sout_gap_next_new", "rout_b3600_new", "sin_b3600_new",
                 "cyc_f_all", "sg_mid7200"]
    for link in ("rout_next", "rout_next_new", "sin_prev", "sin_prev_new"):
        idx = links[link]
        for c in pass_cols:
            v = F[c].to_numpy(float)
            F[f"nb_{link}_{c}"] = np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)
        F[f"nb_{link}_dt"] = np.where(idx >= 0, np.abs(t[np.clip(idx, 0, None)] - t), np.nan)
    print(f"[{split}] all features {F.shape[1]} cols, {time.time() - t0:.0f}s", flush=True)
    os.makedirs("data/cache", exist_ok=True)
    F.to_parquet(path)
    return F


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=600, verbose=-1, n_jobs=1)


def main():
    t0 = time.time()
    tr = features("train")
    te = features("test")
    _, lab, groups = training_frame(["local"])
    tr = tr.set_index("tx").loc[lab.index]
    cols = list(tr.columns)
    X = tr[cols].astype(float)
    y = lab["label"].to_numpy()
    oof = np.zeros(len(y))
    for f, (fit, hold) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=0).split(X, y, groups)):
        m = lgb.LGBMClassifier(**PARAMS, random_state=0)
        m.fit(X.iloc[fit], y[fit])
        oof[hold] = m.predict_proba(X.iloc[hold])[:, 1]
        print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)
    report(oof, lab, "gfp")
    os.makedirs("data/oof", exist_ok=True)
    pd.DataFrame({"tx": lab.index, "oof": oof}).to_parquet("data/oof/gfp.parquet")

    v17 = pd.read_parquet("data/oof/v17_realmotif.parquet").set_index("tx")["oof"].reindex(lab.index)
    ok = v17.notna().to_numpy()
    blend = pd.Series(oof[ok]).rank(pct=True).to_numpy() + v17[ok].rank(pct=True).to_numpy()
    print(f"blend 50/50 rank with v17 OOF ALL AP x100: {100 * average_precision_score(y[ok], blend):.2f} "
          f"(v17 alone {100 * average_precision_score(y[ok], v17[ok]):.2f}, rows {ok.sum()})")
    report(pd.Series(blend).to_numpy(), lab[ok], "blend")

    m = lgb.LGBMClassifier(**PARAMS, random_state=0)
    m.fit(X, y)
    raw = m.predict_proba(te[cols].astype(float))[:, 1]
    imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("top features:", ", ".join(imp.index[:15]))
    write("gfp", te, raw, te["usd_out"].rank().to_numpy())
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
