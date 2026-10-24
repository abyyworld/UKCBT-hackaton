"""Within fast scenarios only (span <= 20 min): which shape features separate easy suspicious from look-alikes."""
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.features import build_split
from src.load import load_labels

txs, X = build_split("train", ["shape"])
lab = load_labels().set_index("tx").loc[txs["tx"]]
lab["ts"] = txs["ts"].to_numpy()
sc = lab[lab.scenario_id.notna()]
span = sc.groupby("scenario_id")["ts"].agg(lambda x: x.max() - x.min())
lab["span"] = lab["scenario_id"].map(span)
m = (lab["span"] <= 1200) & ((lab["difficulty"] == "easy") | (lab["label"] == 0))
print(lab[m].groupby(["label", "family"]).scenario_id.nunique())
y = lab["label"].to_numpy()[m.to_numpy()]
rows = []
for c in X.columns:
    v = X[c].to_numpy()[m.to_numpy()]
    ok = ~pd.isna(v)
    if ok.sum() > 30 and len(set(y[ok])) == 2:
        rows.append((c, roc_auc_score(y[ok], v[ok]), ok.mean()))
r = pd.DataFrame(rows, columns=["f", "auc", "cover"]).assign(d=lambda d: (d.auc - .5).abs()).sort_values("d", ascending=False)
print(r.head(25).to_string())
