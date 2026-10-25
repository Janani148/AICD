"""Data loading for AICD Bench.

HF dataset: AICD-bench/AICD-Bench, configs T1 / T2 / T3, splits train/validation/test,
columns: `code` (str), `label` (int64).

Label ids are NOT documented on the dataset card. The names below follow the order used in
the paper's prompts and are therefore an ASSUMPTION; verify against the data (e.g. inspect a few
samples per label) before reporting per-class results. Macro-F1 itself does not depend on names.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

HF_REPO = "AICD-bench/AICD-Bench"
TASK_CONFIG = {1: "T1", 2: "T2", 3: "T3"}

# Assumed label orders (see module docstring).
LABELS = {
    1: ["human", "machine"],
    2: ["human", "deepseek-ai", "Qwen", "01-ai", "bigcode", "Gemma", "Phi",
        "meta-llama", "ibm-granite", "mistralai", "OpenAI", "Gemini"],
    3: ["human", "machine", "hybrid", "adversarial"],
}


def num_labels(task: int) -> int:
    return len(LABELS[task])


def load_split(task: int, split: str, max_samples: Optional[int] = None,
               seed: int = 0, streaming_head: bool = True):
    """Return (codes: list[str], labels: np.ndarray).

    max_samples: subsample for quick experiments. With streaming_head=True we shuffle a
    streamed buffer instead of downloading the whole (multi-GB) split.
    """
    from datasets import load_dataset

    cfg = TASK_CONFIG[task]
    if max_samples is None:
        ds = load_dataset(HF_REPO, cfg, split=split)
        return list(ds["code"]), np.asarray(ds["label"], dtype=np.int64)

    if streaming_head:
        from tqdm import tqdm
        ds = load_dataset(HF_REPO, cfg, split=split, streaming=True)
        buf = min(10_000, max(max_samples, 1000))
        ds = ds.shuffle(seed=seed, buffer_size=buf)
        codes, labels = [], []
        with tqdm(total=max_samples, desc=f"Streaming {split} (Task {task})", unit="ex") as pbar:
            for ex in ds:
                codes.append(ex["code"])
                labels.append(ex["label"])
                pbar.update(1)
                if len(codes) >= max_samples:
                    break
        del ds
        import gc
        gc.collect()
        return codes, np.asarray(labels, dtype=np.int64)

    ds = load_dataset(HF_REPO, cfg, split=split).shuffle(seed=seed).select(range(max_samples))
    return list(ds["code"]), np.asarray(ds["label"], dtype=np.int64)


def synthetic_split(task: int, n: int = 600, seed: int = 0, shift: bool = False):
    """Tiny offline stand-in used for smoke tests only (NOT a real benchmark)."""
    rng = random.Random(seed)
    k = num_labels(task)
    human_names = ["li", "nums", "pos", "a1", "cur", "tmp", "x"]
    ai_names = ["result", "output", "answer", "index", "tests", "values"]
    codes, labels = [], []
    for i in range(n):
        y = rng.randrange(k)
        names = human_names if y == 0 else ai_names
        a, b = rng.choice(names), rng.choice(names)
        if shift:  # different surface form to mimic a distribution shift
            body = f"function f({a}, {b}) {{ return {a} + {b}; }}"
        elif y == 0:
            body = f"def f({a}, {b}):\n    return {a}+{b}"
        else:
            body = (f"def compute_{a}({a}: int, {b}: int) -> int:\n"
                    f"    \"\"\"Compute the {a}.\"\"\"\n"
                    f"    for i in range({y + 1}):\n        pass\n    return {a} + {b}")
        codes.append(body)
        labels.append(y)
    return codes, np.asarray(labels, dtype=np.int64)
