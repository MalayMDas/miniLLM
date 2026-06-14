"""Run real benchmarks on a checkpoint — all local, no external API.

    python scripts/benchmark.py --ckpt artifacts/ckpt_pretrain/step_0000200.pt \
        --tasks hellaswag,openbookqa,gsm8k --limit 100

Tasks: hellaswag, openbookqa (MCQ log-likelihood), gsm8k (generative exact-match),
bfcl (function-call AST match). VQAv2 needs a multimodal checkpoint + COCO images;
run it via llmscratch.eval.tasks.evaluate_vqa.

First run downloads the dataset from the HF Hub once (cached); scoring is on-device.
For *official, comparable* numbers, the same model wraps into lm-evaluation-harness
(our continuation_logprob == its loglikelihood request).
"""
from __future__ import annotations

# Import pyarrow/datasets BEFORE torch: on Windows, torch-first then a pyarrow load
# segfaults (native DLL clash). Harmless no-op on Linux. Must stay above torch.
try:
    import datasets  # noqa: F401
except Exception:
    pass

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import build_tokenizer
from llmscratch.utils.checkpoint import load_checkpoint
from llmscratch.eval import multiple_choice_accuracy
from llmscratch.eval.tasks import (load_hellaswag, load_openbookqa, load_gsm8k,
                                   evaluate_gsm8k, load_bfcl_jsonl, evaluate_bfcl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tok.json")
    ap.add_argument("--tasks", default="hellaswag,openbookqa,gsm8k,bfcl")
    ap.add_argument("--limit", type=int, default=100, help="examples per task")
    ap.add_argument("--bfcl-file", default="data/sample_bfcl.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer({"mode": "bpe", "path": args.tokenizer})
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Decoder(ModelConfig(**payload["model_config"]))
    load_checkpoint(args.ckpt, model, map_location=device)
    model.to(device).eval()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = {}
    for t in tasks:
        if t == "hellaswag":
            results["hellaswag (acc_norm)"] = multiple_choice_accuracy(
                model, tok, load_hellaswag(limit=args.limit), device=device)
        elif t == "openbookqa":
            results["openbookqa (acc_norm)"] = multiple_choice_accuracy(
                model, tok, load_openbookqa(limit=args.limit), device=device)
        elif t == "gsm8k":
            results["gsm8k (exact)"] = evaluate_gsm8k(
                model, tok, load_gsm8k(limit=args.limit), device=device)
        elif t == "bfcl":
            results["bfcl (ast)"] = evaluate_bfcl(
                model, tok, load_bfcl_jsonl(args.bfcl_file, limit=args.limit), device=device)
        else:
            print(f"  (skipping unknown/multimodal task: {t})")

    print(f"\n=== benchmarks: {Path(args.ckpt).name} (limit={args.limit}) ===")
    for k, v in results.items():
        print(f"  {k:26}: {v:.4f}")


if __name__ == "__main__":
    main()
