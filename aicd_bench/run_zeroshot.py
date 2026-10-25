"""Zero-shot baselines.

  python run_zeroshot.py --method fastdetect --task 1 --max-test 2000
  python run_zeroshot.py --method llm --task 3 --cot --max-test 2000     # needs GOOGLE_API_KEY
"""
import argparse
import os

from aicd import data, metrics
from aicd.zeroshot import FastDetectGPT, llm_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fastdetect", "llm"], required=True)
    ap.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--llm-model", default="gemini-2.5-flash")
    ap.add_argument("--scoring-model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--max-test", type=int, default=2000, help="paper used a stratified 5% sample")
    ap.add_argument("--calib", type=int, default=1000, help="validation samples to set threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    Xte, yte = data.load_split(a.task, "test", a.max_test, a.seed)
    os.makedirs(a.out, exist_ok=True)

    if a.method == "fastdetect":
        if a.task == 2:
            raise SystemExit("Fast-DetectGPT is binary: use task 1, or task 3 (human vs rest).")
        det = FastDetectGPT(a.scoring_model)
        Xv, yv = data.load_split(a.task, "validation", a.calib, a.seed)
        det.fit_threshold(Xv, (yv != 0).astype(int))      # label 0 = human (assumed)
        pred = det.predict(Xte)
        score = metrics.macro_f1((yte != 0).astype(int), pred)
        name = f"task{a.task}_fastdetectgpt"
        res = {"macro_f1": score, "threshold": det.threshold}
    else:
        pred = llm_predict(a.task, Xte, a.cot, a.llm_model)
        score = metrics.macro_f1(yte, pred)
        name = f"task{a.task}_llm_{'cot' if a.cot else 'zs'}"
        res = metrics.full_report(yte, pred, data.LABELS[a.task])

    metrics.save_json(res, f"{a.out}/{name}.json")
    print(f"{name}: Macro-F1={score:.2f}")


if __name__ == "__main__":
    main()
