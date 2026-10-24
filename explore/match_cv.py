"""Mean out-of-fold AP over seeds 0,1,2 for module sets given as arguments (comma separated)."""
import os
import sys
import numpy as np
from src.cv import oof_scores, report, training_frame

for mods in [m.split(",") for m in sys.argv[1:]]:
    X, lab, groups = training_frame(mods)
    y = lab["label"].to_numpy()
    res = []
    for seed in [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]:
        oof = oof_scores(X, y, groups, seed=seed)
        report(oof, lab, f"{','.join(mods)} seed {seed}")
        sys.stdout.flush()
        from sklearn.metrics import average_precision_score as aps
        r = [aps(y, oof)] + [aps(y[m], oof[m]) for m in
                             (lab["difficulty"].isin([d, "normal"]).to_numpy() for d in ("easy", "medium", "hard"))]
        res.append(r)
    a = 100 * np.array(res)
    print(",".join(mods), "seeds", np.round(a[:, 0], 2),
          "mean ALL %.2f easy %.2f medium %.2f hard %.2f" % tuple(a.mean(0)), flush=True)
