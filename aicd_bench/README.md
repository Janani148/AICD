# AICD Bench – baseline implementation

Re-implementation of the evaluation suite described in *"AICD Bench: A Challenging Benchmark for
AI-Generated Code Detection"* (EACL 2026). Data: <https://huggingface.co/datasets/AICD-bench/AICD-Bench>
(configs `T1`, `T2`, `T3`; columns `code`, `label`).

| Task | Config | Classes | Train / Val / Test |
|---|---|---|---|
| 1 Robust binary (human vs machine, OOD) | `T1` | 2 | 500K / 100K / 1.1M |
| 2 Model-family attribution | `T2` | 12 | 502K / 101K / 508K |
| 3 Fine-grained (human/machine/hybrid/adversarial) | `T3` | 4 | 900K / 200K / 1M |

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
# Classical baselines (LR / SVM / CatBoost x tfidf / ast / both)
python run_classical.py --task 1 --features tfidf --clf svm --max-train 100000 --max-test 100000

# Encoder fine-tuning (codebert, codet5p, unixcoder, modernbert, roberta, deberta); GPU recommended
python run_neural.py --task 3 --model modernbert --max-train 200000 --max-test 100000

# Zero-shot: Fast-DetectGPT (tasks 1, 3-binarised) and prompted LLMs (zero-shot / CoT)
python run_zeroshot.py --method fastdetect --task 1
GOOGLE_API_KEY=... python run_zeroshot.py --method llm --task 3 --cot

# Offline smoke test with toy data
python run_classical.py --task 1 --features both --clf lr --synthetic
```
Results (Macro-F1 x100, per-class report, row-normalised confusion matrix) go to `results/*.json`.
Drop `--max-*` flags to use full splits (multi-GB download; neural training on the full 500K-900K
train sets takes many GPU-hours).

## Protocol (from the paper)
* Metric: Macro-F1. Random baseline samples from train priors; majority baseline predicts the top class.
* TF-IDF: whitespace tokens, no normalisation, 1–3-grams, 5,000 features -> SVD 500.
* AST features: Tree-Sitter depth / node-type frequencies / layout stats (dimension is data-driven).
* Encoders: 3 epochs, batch 64, 512 tokens.
* Zero-shot LLMs: Gemini-2.5-Flash and Gemma3-27B on a stratified 5% sample; prompts in `aicd/zeroshot.py`.

## Caveats – please read
* **Label names are assumed.** The dataset card lists only integer labels. `aicd/data.py::LABELS` uses
  the order from the paper's prompts (human=0 ...). Verify on samples before trusting per-class names.
* **No language/domain columns** are in the HF schema, so Task 1's four seen/unseen splits and Task 3's
  in/out-of-domain split cannot be reconstructed exactly; `run_task1_breakdown.py` gives only a
  heuristic per-language proxy. Check the dataset viewer for extra columns / ordering conventions.
* My AST features are my own approximation of the paper's (exact feature list isn't published in the text);
  expect numbers to differ from Table 4.
* Fast-DetectGPT uses a single scoring model (default Qwen2.5-Coder-1.5B) with a threshold calibrated on
  validation data; the paper does not specify its scoring model.
* CodeT5+ has no stock sequence-classification head in the `-embedding` checkpoint; see note in `aicd/neural.py`.
* Not verified against the real dataset: only the offline synthetic smoke test was run.
