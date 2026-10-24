"""Mean out-of-fold AP over seeds 0,1,2: local vs local+shape."""
import sys
import numpy as np
from src.cv import oof_scores, training_frame
from sklearn.metrics import average_precision_score

for mods in (["local"], ["local", "shape"]) if len(sys.argv) < 2 else [m.split(",") for m in sys.argv[1:]]:
    X, lab, groups = training_frame(mods)
    y = lab["label"].to_numpy()
    res = []
    for seed in (0, 1, 2):
        oof = oof_scores(X, y, groups, seed=seed)
        r = [average_precision_score(y, oof)]
        for diff in ("easy", "medium"):
            m = lab["difficulty"].isin([diff, "normal"]).to_numpy()
            r.append(average_precision_score(y[m], oof[m]))
        res.append(r)
    a = 100 * np.array(res)
    print(",".join(mods), "seeds", np.round(a[:, 0], 2), "mean ALL %.2f easy %.2f medium %.2f" % tuple(a.mean(0)), flush=True)
