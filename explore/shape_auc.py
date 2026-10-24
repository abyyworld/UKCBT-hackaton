"""AUC of each shape feature: easy suspicious vs look-alike normal (label 0 with scenario_id). Analysis only."""
import time
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.features import build_split
from src.load import load_labels

t0 = time.time()
txs, X = build_split("train", ["shape"])
print("built", X.shape, round(time.time() - t0), "s")
lab = load_labels().set_index("tx").loc[txs["tx"]]
m = ((lab["difficulty"] == "easy") | ((lab["label"] == 0) & lab["scenario_id"].notna() & ~lab["family"].str.contains("relay"))).to_numpy()
y = lab["label"].to_numpy()[m]
rows = []
for c in X.columns:
    v = X[c].to_numpy()[m]
    ok = ~pd.isna(v)
    if ok.sum() > 50 and len(set(y[ok])) == 2:
        rows.append((c, roc_auc_score(y[ok], v[ok]), ok.mean()))
r = pd.DataFrame(rows, columns=["f", "auc", "cover"]).assign(d=lambda d: (d.auc - .5).abs()).sort_values("d", ascending=False)
print(r.head(30).to_string())
