"""Shift audit: which features move between train and test on motif-like rows, and in which direction.

    python -m explore.shift_audit           (needs features cached by explore/shift_audit_build.py)

Motif-like rows: episode all300 has 4 to 60 payments and most of them are first-time pairs. For every feature:
- ks: KS statistic train motif rows vs test motif rows (same selection in both splits)
- sep: (train suspicious median - train normal median) in units of the train IQR (signed)
- shift: (test median - train median) in units of (train suspicious median - train normal median).
  Negative = test moves toward the train normal class, so test laundering would look more normal to the model.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features import build_split
from src.load import load_labels

MODS = "local,episodes,isolation,patterns,shape,slow".split(",")


def motif_rows(X):
    size = X["episodes__all300_size"]
    return ((size >= 4) & (size <= 60) & (X["episodes__all300_new_share"] >= 0.5)).to_numpy()


if __name__ == "__main__":
    tr_txs, Xtr = build_split("train", MODS)
    te_txs, Xte = build_split("test", MODS)
    y = load_labels().set_index("tx").loc[tr_txs["tx"], "label"].to_numpy()
    mtr, mte = motif_rows(Xtr), motif_rows(Xte)
    print(f"motif rows: train {mtr.sum()} ({y[mtr].mean():.1%} suspicious), test {mte.sum()}")
    rows = []
    for c in Xtr.columns:
        a, b = Xtr.loc[mtr, c].to_numpy(float), Xte.loc[mte, c].to_numpy(float)
        sp, nm = a[y[mtr] == 1], a[y[mtr] == 0]
        sp, nm, a, b = (v[~np.isnan(v)] for v in (sp, nm, a, b))
        if len(sp) < 30 or len(nm) < 30 or len(b) < 30 or np.nanstd(a) == 0:
            continue
        ms, mn, mtrn, mtst = np.median(sp), np.median(nm), np.median(a), np.median(b)
        iqr = np.subtract(*np.percentile(a, [75, 25])) or np.std(a) or 1.0
        gap = ms - mn
        rows.append(dict(feature=c, ks=ks_2samp(a, b).statistic,
                         class_ks=ks_2samp(sp, nm).statistic, sep=gap / iqr,
                         shift=(mtst - mtrn) / gap if abs(gap) > 1e-9 else np.nan,
                         susp=ms, normal=mn, train=mtrn, test=mtst,
                         nan_tr=np.isnan(Xtr.loc[mtr, c].to_numpy(float)).mean(),
                         nan_te=np.isnan(Xte.loc[mte, c].to_numpy(float)).mean()))
    R = pd.DataFrame(rows)
    # risk: big train/test shift on a feature that separates the classes, moving toward normal
    R["risk"] = R["ks"] * R["class_ks"] * (R["shift"] < 0)
    R = R.sort_values("risk", ascending=False)
    R.to_csv("data/shift_audit.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.head(40).round(3).to_string(index=False))
    print("\nlargest KS overall")
    print(R.sort_values("ks", ascending=False).head(25).round(3).to_string(index=False))
