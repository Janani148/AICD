"""Task 1 breakdown by language (heuristic guess) -- the dataset has no language/domain column,
so the paper's four seen/unseen splits cannot be rebuilt from `code`+`label` alone. This script
reports Macro-F1 per guessed language for any saved predictions file (.npy) as a proxy.

  python run_task1_breakdown.py --pred preds.npy --max-test 50000
"""
import argparse

import numpy as np
from sklearn.metrics import f1_score

from aicd import data
from aicd.features import guess_language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    X, y = data.load_split(1, "test", a.max_test, a.seed)
    p = np.load(a.pred)
    langs = np.array([guess_language(c) for c in X])
    for l in sorted(set(langs)):
        m = langs == l
        print(f"{l:12s} n={m.sum():7d} macroF1={100 * f1_score(y[m], p[m], average='macro'):.2f}")


if __name__ == "__main__":
    main()
