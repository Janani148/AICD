"""Fine-tune an encoder and evaluate with Macro-F1.

  python run_neural.py --task 3 --model modernbert --max-train 100000 --max-test 50000
"""
import argparse
import os

import torch

from aicd import data, metrics
from aicd.neural import ENCODERS, build_model, predict, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--model", default="modernbert", help=f"one of {list(ENCODERS)} or an HF path")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr = data.load_split(a.task, "train", a.max_train, a.seed)
    Xte, yte = data.load_split(a.task, "test", a.max_test, a.seed)

    tok, model = build_model(a.model, data.num_labels(a.task))
    model = train(model, tok, Xtr, ytr, a.epochs, a.batch_size, a.max_len, a.lr, device)
    pred = predict(model, tok, Xte, a.max_len, a.batch_size, device)

    res = metrics.full_report(yte, pred, data.LABELS[a.task])
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/task{a.task}_{a.model}.json"
    metrics.save_json(res, path)
    print(f"task{a.task} {a.model}: Macro-F1={res['macro_f1']:.2f} -> {path}")


if __name__ == "__main__":
    main()
