"""Feature registry. Each module exposes build(txs) -> DataFrame indexed by tx, without reading labels.

Features are cached per split in data/cache/ so modules can be iterated on quickly.
"""
import hashlib
import importlib
import os

import pandas as pd

from src.load import load_split

MODULES = ["local"]
CACHE = "data/cache"


def build_split(split, modules=None, refresh=False):
    modules = modules or MODULES
    txs, _ = load_split(split)
    parts = []
    for name in modules:
        src = open(os.path.join(os.path.dirname(__file__), f"{name}.py"), "rb").read()
        version = hashlib.md5(src).hexdigest()[:8]  # cache refreshes whenever the module code changes
        path = f"{CACHE}/{split}_{name}_{version}.parquet"
        if os.path.exists(path) and not refresh:
            part = pd.read_parquet(path)
        else:
            part = importlib.import_module(f"src.features.{name}").build(txs)
            part = part.add_prefix(f"{name}__")
            os.makedirs(CACHE, exist_ok=True)
            part.to_parquet(path)
        parts.append(part)
    X = pd.concat(parts, axis=1)
    return txs, X.loc[txs["tx"].to_numpy()]
