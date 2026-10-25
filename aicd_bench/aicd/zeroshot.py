"""Zero-shot baselines: Fast-DetectGPT (binary) and prompted LLMs (zero-shot / CoT)."""
from __future__ import annotations

import os
import re
from typing import List

import numpy as np

# ------------------------------------------------------------------ Fast-DetectGPT
class FastDetectGPT:
    """Analytic conditional-probability-curvature criterion (Bao et al., ICLR'24).

    Uses one scoring model for both sampling and reference (the 'same model' variant);
    pass a second model pair for the original two-model setting. Higher score => more
    machine-like. Threshold is calibrated on a labelled dev set via `fit_threshold`.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B", device="cuda", max_len=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
        ).to(device).eval()
        self.device, self.max_len, self.threshold = device, max_len, 0.0

    def score(self, code: str) -> float:
        torch = self.torch
        enc = self.tok(code, return_tensors="pt", truncation=True,
                       max_length=self.max_len).to(self.device)
        if enc.input_ids.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            logits = self.model(**enc).logits[:, :-1].float()
        labels = enc.input_ids[:, 1:]
        lprobs = torch.log_softmax(logits, dim=-1)
        probs = lprobs.exp()
        ll = lprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mean_ref = (probs * lprobs).sum(-1)
        var_ref = (probs * lprobs.square()).sum(-1) - mean_ref.square()
        return float(((ll.sum(-1) - mean_ref.sum(-1)) / var_ref.sum(-1).clamp_min(1e-8).sqrt()).item())

    def scores(self, codes: List[str]) -> np.ndarray:
        return np.array([self.score(c) for c in codes])

    def fit_threshold(self, codes, labels_binary):
        """labels_binary: 1 = machine/non-human. Picks the threshold maximising macro-F1."""
        from sklearn.metrics import f1_score
        s = self.scores(codes)
        cands = np.quantile(s, np.linspace(0.02, 0.98, 97))
        best = max(cands, key=lambda t: f1_score(labels_binary, s > t, average="macro"))
        self.threshold = float(best)
        return self.threshold

    def predict(self, codes):
        return (self.scores(codes) > self.threshold).astype(int)


# ------------------------------------------------------------------ prompted LLMs
PROMPTS_ZS = {
    1: "Reply with exactly one word Human or AI.\nGiven this code, identify its origin\n{code}",
    2: ("Reply with exactly one word from this list:\nhuman, deepseek-ai, Qwen, 01-ai, bigcode, "
        "Gemma, Phi, meta-llama, ibm-granite, mistralai, OpenAI, Gemini.\n"
        "Given this code, identify its origin\n{code}"),
    3: ("Reply with exactly one word Human, AI, Hybrid or Adversarial.\n"
        "Given this code, was it written by Human, AI, in Hybrid collaboration or by an "
        "Adversarial model which tried to fool the detector mimicing human?\n{code}"),
}

_COT_SIGNALS = """Consider (but do not list) signals such as:
- Naming: verbosity, camelCase/snake_case consistency, oddly generic names.
- Comments/docstrings: density, template-like phrasing, per-function summaries.
- Type hints: pervasive and consistent vs ad hoc.
- Structure/style: formatter fingerprints, exhaustive edge cases vs idiosyncratic style, TODO markers.
- Boilerplate: license blocks, scaffolds, placeholder names, auto-generated patterns.
- AI artifacts: tutorial-like narration, code mirroring common LLM examples.
- Human artifacts: partial implementations, inline hacks, commented-out experiments.
If uncertain, choose the most likely origin using the balance of signals."""

PROMPTS_COT = {
    1: ("You are a code-origin classifier. Infer whether a code snippet was written by a Human "
        "or an AI assistant. Reason silently; do NOT reveal reasoning. Output exactly one word: "
        "Human or AI.\n" + _COT_SIGNALS + "\nCode:\n```{code}```"),
    2: ("You are a code-origin classifier. Infer which single author produced the code: human, "
        "deepseek-ai, Qwen, 01-ai, bigcode, Gemma, Phi, meta-llama, ibm-granite, mistralai, "
        "OpenAI, Gemini. Think silently; output exactly one label from that list.\n"
        + _COT_SIGNALS + "\nCode:\n```{code}```"),
    3: ("You are a code-origin classifier. Infer whether a snippet was written by a Human, an AI "
        "assistant, Hybridly generated, or by an Adversarial model trained to mimic humans. "
        "Reason silently; output exactly one word: Human, AI, Hybrid, or Adversarial.\n"
        + _COT_SIGNALS + "\nCode:\n```{code}```"),
}

_PARSE = {
    1: {"human": 0, "ai": 1},
    2: {n.lower(): i for i, n in enumerate(
        ["human", "deepseek-ai", "Qwen", "01-ai", "bigcode", "Gemma", "Phi", "meta-llama",
         "ibm-granite", "mistralai", "OpenAI", "Gemini"])},
    3: {"human": 0, "ai": 1, "hybrid": 2, "adversarial": 3},
}


def parse_answer(task: int, text: str) -> int:
    text = text.strip().lower()
    # longest-first so "01-ai" is not shadowed by shorter labels
    for name in sorted(_PARSE[task], key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text):
            return _PARSE[task][name]
    return 0  # unparseable -> counts as an error for non-human items


def llm_predict(task: int, codes: List[str], cot: bool, model: str = "gemini-2.5-flash",
                backend: str = "gemini", max_code_chars: int = 6000, call=None) -> np.ndarray:
    """backend='gemini' needs GOOGLE_API_KEY. Pass `call(prompt)->str` to use any other LLM
    (e.g. a local Gemma3-27B served by vLLM/ollama)."""
    tpl = (PROMPTS_COT if cot else PROMPTS_ZS)[task]
    if call is None:
        if backend != "gemini":
            raise ValueError("provide `call` for non-gemini backends")
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        call = lambda p: client.models.generate_content(model=model, contents=p).text or ""
    preds = []
    for c in codes:
        try:
            out = call(tpl.format(code=c[:max_code_chars]))
        except Exception:
            out = ""
        preds.append(parse_answer(task, out))
    return np.array(preds)
