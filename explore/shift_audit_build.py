"""Build and cache features for train and test (for the shift audit)."""
import sys
from src.features import build_split

MODS = "local,episodes,isolation,patterns,shape,slow".split(",")
if __name__ == "__main__":
    for split in sys.argv[1:]:
        txs, X = build_split(split, MODS)
        print(split, X.shape)
