"""Train/evaluate classical baselines.

  python run_classical.py --task 1 --features tfidf --clf svm --max-train 50000 --max-test 50000
  python run_classical.py --task 3 --features both --clf catboost --synthetic   # offline smoke test
"""
import argparse
import os
import time

from aicd import data, metrics
from aicd.classical import make_classifier
from aicd.features import CombinedFeaturizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--features", default="tfidf", choices=["tfidf", "ast", "both"])
    ap.add_argument("--clf", default="svm", choices=["lr", "svm", "catboost"])
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true", help="offline toy data (smoke test only)")
    a = ap.parse_args()

    if a.synthetic:
        Xtr, ytr = data.synthetic_split(a.task, a.max_train or 600, a.seed)
        Xte, yte = data.synthetic_split(a.task, a.max_test or 300, a.seed + 1, shift=True)
    else:
        Xtr, ytr = data.load_split(a.task, "train", a.max_train, a.seed)
        Xte, yte = data.load_split(a.task, "test", a.max_test, a.seed)

    t0 = time.time()
    feat = CombinedFeaturizer(a.features, svd_dim=500 if not a.synthetic else 20)
    Ftr = feat.fit_transform(Xtr)
    Fte = feat.transform(Xte)
    clf = make_classifier(a.clf, a.seed).fit(Ftr, ytr)
    pred = clf.predict(Fte).ravel().astype(int)

    names = data.LABELS[a.task]
    res = metrics.full_report(yte, pred, names if len(set(yte)) <= len(names) else None)
    res["dummy_baselines"] = metrics.baselines(ytr, yte, len(names), a.seed)
    res["seconds"] = time.time() - t0
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/task{a.task}_{a.clf}_{a.features}.json"
    metrics.save_json(res, path)
    print(f"task{a.task} {a.clf}/{a.features}: Macro-F1={res['macro_f1']:.2f} "
          f"(random={res['dummy_baselines']['random']:.2f}, "
          f"majority={res['dummy_baselines']['majority']:.2f}) -> {path}")


if __name__ == "__main__":
    main()
