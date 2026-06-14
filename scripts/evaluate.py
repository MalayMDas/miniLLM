"""Evaluate a checkpoint against benchmarks (perplexity + multiple-choice).

    python scripts/evaluate.py --ckpt artifacts/ckpt_pretrain/step_0000200.pt

Runs our transparent harness so the numbers are explainable. For *official,
comparable* benchmark scores (HellaSwag / ARC / MMLU / GSM8K), wire the same model
into lm-evaluation-harness — the loglikelihood primitive in eval/scoring.py is
exactly what it expects, so it's a thin adapter, not a rewrite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import build_tokenizer
from llmscratch.utils.checkpoint import load_checkpoint
from llmscratch.eval import perplexity, multiple_choice_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tok.json")
    ap.add_argument("--ppl-file", default="data/sample.txt")
    ap.add_argument("--mcq", default="data/sample_mcq.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer({"mode": "bpe", "path": args.tokenizer})
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Decoder(ModelConfig(**payload["model_config"]))
    load_checkpoint(args.ckpt, model, map_location=device)
    model.to(device).eval()

    results = {}
    if args.ppl_file and Path(args.ppl_file).exists():
        texts = [l.strip() for l in open(args.ppl_file, encoding="utf-8") if l.strip()]
        results["perplexity"] = perplexity(model, tok, texts, device=device)
    if args.mcq and Path(args.mcq).exists():
        examples = [json.loads(l) for l in open(args.mcq, encoding="utf-8") if l.strip()]
        results["mcq_accuracy"] = multiple_choice_accuracy(model, tok, examples, device=device)

    print(f"\n=== eval: {Path(args.ckpt).name} ===")
    for k, v in results.items():
        print(f"  {k:16}: {v:.4f}")
    return results


if __name__ == "__main__":
    main()
