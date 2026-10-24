"""Canonical tables every module works with.

`load.py` turns the organiser files into these three tables; everything downstream
(features, training, prediction) only ever sees this shape.

txs     one row per transaction (all splits)
    tx          str        TRANSACTION_HASH
    split       str        "train" or "test"
    ts          datetime   UTC close time
    sender      str        the transaction's Account
    receiver    str        final destination (Destination)
    tx_type     str        e.g. "Payment"
    n_legs      int        number of transfer rows (1 to 3)
    asset_in    str        asset the sender paid ("XRP" or "SYMBOL:ISSUER")
    asset_out   str        asset the receiver got
    amount_in   float      units of asset_in actually paid (first leg)
    amount_out  float      units of asset_out delivered (last leg)
    usd_in      float      USD value of amount_in at the payment's hour
    usd_out     float      USD value of amount_out
    is_conversion bool     asset_in != asset_out
    fee_xrp     float      fee burned, in XRP
    dest_tag    float      destination tag, NaN when absent
    success     bool       transaction applied (tesSUCCESS)

edges   one row per transfer leg (money movement)
    tx, split, ts, leg (0-based), src, dst, asset, amount, usd

labels  training split only
    tx      str
    label   int  1 = suspicious
"""

TXS_COLS = [
    "tx", "split", "ts", "sender", "receiver", "tx_type", "n_legs",
    "asset_in", "asset_out", "amount_in", "amount_out", "usd_in", "usd_out",
    "is_conversion", "fee_xrp", "dest_tag", "success",
]
EDGES_COLS = ["tx", "split", "ts", "leg", "src", "dst", "asset", "amount", "usd"]
LABELS_COLS = ["tx", "label"]
SPLITS = ("train", "test")


def check(txs, edges, labels=None):
    """Fail fast if a table drifts from the contract."""
    for name, df, cols in (("txs", txs, TXS_COLS), ("edges", edges, EDGES_COLS)):
        missing = [c for c in cols if c not in df.columns]
        assert not missing, f"{name} missing columns: {missing}"
    assert txs["tx"].is_unique, "txs.tx must be unique"
    assert set(edges["tx"]) <= set(txs["tx"]), "edges reference unknown tx"
    if labels is not None:
        assert set(labels["tx"]) <= set(txs.loc[txs["split"] == "train", "tx"]), "labels outside train"
