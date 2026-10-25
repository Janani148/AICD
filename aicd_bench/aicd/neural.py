"""Encoder fine-tuning (paper setup: 3 epochs, batch 64, 512-token window)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

ENCODERS = {
    "codebert": "microsoft/codebert-base",
    "codet5p": "Salesforce/codet5p-110m-embedding",  # see note in build_model
    "unixcoder": "microsoft/unixcoder-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "roberta": "FacebookAI/roberta-base",
    "deberta": "microsoft/deberta-v3-base",
}


class CodeDS(Dataset):
    def __init__(self, codes: List[str], labels: Optional[np.ndarray], tok, max_len: int):
        self.codes, self.labels, self.tok, self.max_len = codes, labels, tok, max_len

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        enc = self.tok(self.codes[i], truncation=True, max_length=self.max_len)
        if self.labels is not None:
            enc["labels"] = int(self.labels[i])
        return enc


def _collate(tok):
    def fn(batch):
        labels = [b.pop("labels") for b in batch if "labels" in b]
        out = tok.pad(batch, return_tensors="pt")
        if labels:
            out["labels"] = torch.tensor(labels)
        return out
    return fn


def build_model(name: str, n_labels: int):
    path = ENCODERS.get(name, name)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    # NOTE: CodeT5+ ships an embedding-only checkpoint; if sequence classification fails to
    # load for it, wrap its encoder with a linear head or pass a different checkpoint path.
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=n_labels, trust_remote_code=True)
    return tok, model


@torch.no_grad()
def predict(model, tok, codes, max_len=512, batch_size=64, device="cuda"):
    model.eval()
    dl = DataLoader(CodeDS(codes, None, tok, max_len), batch_size=batch_size,
                    collate_fn=_collate(tok))
    preds = []
    for batch in tqdm(dl, desc="predict"):
        batch = {k: v.to(device) for k, v in batch.items()}
        preds.append(model(**batch).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def train(model, tok, codes, labels, epochs=3, batch_size=64, max_len=512, lr=2e-5,
          device="cuda", fp16=True):
    model.to(device)
    dl = DataLoader(CodeDS(codes, labels, tok, max_len), batch_size=batch_size,
                    shuffle=True, collate_fn=_collate(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    use_amp = fp16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for ep in range(epochs):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep + 1}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            pbar.set_postfix(loss=float(loss.detach().item()))
    return model
