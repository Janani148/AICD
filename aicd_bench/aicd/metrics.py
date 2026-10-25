from __future__ import annotations

import json
from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def macro_f1(y_true, y_pred) -> float:
    """Primary metric of the benchmark (x100 to match the paper's tables)."""
    return 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)


def full_report(y_true, y_pred, label_names: Sequence[str] | None = None) -> dict:
    labels = list(range(len(label_names))) if label_names else None
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=label_names,
            zero_division=0, output_dict=True),
        "confusion_row_normalized": (
            confusion_matrix(y_true, y_pred, labels=labels, normalize="true").tolist()),
    }


def baselines(y_train, y_test, n_classes: int, seed: int = 0) -> dict:
    """Random (prior-sampled) and majority baselines as in Table 4."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum()
    rand = rng.choice(classes, size=len(y_test), p=probs)
    maj = np.full(len(y_test), classes[counts.argmax()])
    return {"random": macro_f1(y_test, rand), "majority": macro_f1(y_test, maj)}


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
