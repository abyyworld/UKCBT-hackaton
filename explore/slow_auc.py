"""AUC and AP of each slow feature: medium suspicious vs all normal, and vs look-alikes. Analysis only."""
import sys
import time
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.features import build_split
from src.load import load_labels

mod = sys.argv[1] if len(sys.argv) > 1 else "slow"
t0 = time.time()
txs, X = build_split("train", [mod])
print("built", X.shape, round(time.time() - t0), "s")
lab = load_labels().set_index("tx").loc[txs["tx"]]
m = lab["difficulty"].isin(["medium", "normal"]).to_numpy()
y = lab["label"].to_numpy()[m]
rows = []
for c in X.columns:
    v = X[c].to_numpy()[m]
    ok = ~pd.isna(v)
    auc = roc_auc_score(y[ok], v[ok])
    sign = 1 if auc >= .5 else -1
    vv = pd.Series(v).fillna(pd.Series(v).min() - 1 if sign > 0 else pd.Series(v).max() + 1).to_numpy()
    rows.append((c, auc, 100 * average_precision_score(y, sign * vv), ok.mean()))
print(pd.DataFrame(rows, columns=["f", "auc", "ap", "cover"]).sort_values("ap", ascending=False).to_string())
