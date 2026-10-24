"""Turn an organiser split directory into the canonical txs / edges tables (see schema.py).

    python -m src.load <split_dir> <split_name>    # quick summary

Splits used: train (labelled), validation (unlabelled practice), test (scored).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

XRP_SENTINEL = "0x1111222211112222111122221111222211112222"

TRAIN_DIR = Path("data/decode-and-detect-training-data/train")
VALID_DIR = Path("data/decode-and-detect-training-data/validation")
TEST_DIR = Path("data/decode-and-detect-test-data")
SPLIT_DIRS = {"train": TRAIN_DIR, "validation": VALID_DIR, "test": TEST_DIR}


def asset_key(token_address, symbol):
    return np.where(token_address == XRP_SENTINEL, "XRP", token_address + ":" + symbol)


def _rates(directory):
    r = pd.read_csv(directory / "market_rates.csv")
    return {(int(t), a): p / 1e6 for t, a, p in zip(r.TIMESTAMP, r.ASSET, r.PRICE_MICRO_USD)}


def _usd(rates, ts, assets, amounts):
    hours = (ts // 3600 * 3600).astype(np.int64)
    return np.array([a * rates[(h, k)] for h, k, a in zip(hours, assets, amounts)])


def load_split(name, directory=None):
    directory = Path(directory or SPLIT_DIRS[name])
    rates = _rates(directory)
    t = pd.read_parquet(directory / "transactions.parquet")
    f = pd.read_parquet(directory / "transfers.parquet")

    ts = t["TIMESTAMP"].astype("int64").to_numpy()
    extra = t["EXTRA_TRANSACTION_FIELDS"].map(json.loads)
    delivered = t["DELIVERED_AMOUNT"].map(json.loads)
    out_asset = delivered.map(lambda d: "XRP" if isinstance(d, str) else f"{d['issuer']}:{d['currency']}")
    out_amount = delivered.map(lambda d: int(d) / 1e6 if isinstance(d, str) else float(d["value"]))

    f = f.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    f["asset"] = asset_key(f["TOKEN_ADDRESS"].to_numpy(), f["SYMBOL"].to_numpy())
    f["amount"] = f["VALUE"].astype(float)
    f["ts"] = f["TIMESTAMP"].astype("int64")
    f["usd"] = _usd(rates, f["ts"].to_numpy(), f["asset"].to_numpy(), f["amount"].to_numpy())
    first = f.groupby("TRANSACTION_HASH").first()
    legs = f.groupby("TRANSACTION_HASH").size()

    txs = pd.DataFrame({
        "tx": t["TRANSACTION_HASH"].to_numpy(),
        "split": name,
        "ts": ts,
        "sender": t["ACCOUNT"].to_numpy(),
        "receiver": extra.map(lambda e: e["Destination"]).to_numpy(),
        "tx_type": t["TRANSACTION_TYPE"].to_numpy(),
        "asset_out": out_asset.to_numpy(),
        "amount_out": out_amount.to_numpy(),
        "fee_xrp": t["FEE"].astype(float).to_numpy(),
        "dest_tag": extra.map(lambda e: e.get("DestinationTag", np.nan)).astype(float).to_numpy(),
        "success": t["SUCCESS"].astype(bool).to_numpy(),
        "seq": t["SEQUENCE"].astype("int64").to_numpy(),
        "ledger": t["LEDGER_INDEX"].astype("int64").to_numpy(),
        "tx_index": t["TRANSACTION_INDEX"].astype("int64").to_numpy(),
        "last_ledger_gap": (pd.to_numeric(t["LAST_LEDGER_SEQUENCE"], errors="coerce") - t["LEDGER_INDEX"].astype("int64")).to_numpy(),
        "flags": pd.to_numeric(t["FLAGS"], errors="coerce").to_numpy(),
        "has_memo": t["MEMOS"].notna().to_numpy(),
        "source_tag": pd.to_numeric(t["SOURCE_TAG"], errors="coerce").to_numpy(),
        "send_max_set": extra.map(lambda e: "SendMax" in e).to_numpy(),
    })
    txs["asset_in"] = txs["tx"].map(first["asset"])
    txs["amount_in"] = txs["tx"].map(first["amount"])
    txs["usd_in"] = txs["tx"].map(first["usd"])
    txs["n_legs"] = txs["tx"].map(legs).astype(int)
    txs["usd_out"] = _usd(rates, ts, txs["asset_out"].to_numpy(), txs["amount_out"].to_numpy())
    txs["is_conversion"] = txs["asset_in"] != txs["asset_out"]

    edges = pd.DataFrame({
        "tx": f["TRANSACTION_HASH"].to_numpy(), "split": name, "ts": f["ts"].to_numpy(),
        "leg": f["TRANSFER_INDEX"].astype(int).to_numpy(), "src": f["FROM_ADDRESS"].to_numpy(),
        "dst": f["TO_ADDRESS"].to_numpy(), "asset": f["asset"].to_numpy(),
        "amount": f["amount"].to_numpy(), "usd": f["usd"].to_numpy(),
    })
    return txs, edges


def load_labels():
    lab = pd.read_csv(TRAIN_DIR / "labels.csv")
    return lab.rename(columns={"TRANSACTION_HASH": "tx", "is_suspicious": "label"})


if __name__ == "__main__":
    txs, edges = load_split(sys.argv[2], sys.argv[1])
    print(txs.shape, edges.shape)
    print(txs.head().to_string())
