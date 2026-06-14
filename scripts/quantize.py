"""Quantize a checkpoint and report the size/quality trade-off.

    python scripts/quantize.py --ckpt artifacts/ckpt_pretrain/step_0000200.pt

Loads fp32 weights, measures size + perplexity, applies dynamic int8, and reports
both again so the trade-off is explicit (best practice: never quote size savings
without the quality delta).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import build_tokenizer
from llmscratch.utils.checkpoint import load_checkpoint
from llmscratch.eval import perplexity
from llmscratch.quantize import quantize_dynamic_int8, serialized_size_bytes

EVAL_TEXT = ["the quick brown fox", "language models predict the next token",
             "attention is all you need"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tok.json")
    args = ap.parse_args()

    tok = build_tokenizer({"mode": "bpe", "path": args.tokenizer})
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Decoder(ModelConfig(**payload["model_config"]))
    load_checkpoint(args.ckpt, model, map_location="cpu")
    model.eval()

    fp32_mb = serialized_size_bytes(model) / 1e6
    fp32_ppl = perplexity(model, tok, EVAL_TEXT, device="cpu")

    qmodel = quantize_dynamic_int8(model)
    int8_mb = serialized_size_bytes(qmodel) / 1e6
    int8_ppl = perplexity(qmodel, tok, EVAL_TEXT, device="cpu")

    print(f"{'':12}{'size (MB)':>12}{'perplexity':>14}")
    print(f"{'fp32':12}{fp32_mb:12.2f}{fp32_ppl:14.3f}")
    print(f"{'int8':12}{int8_mb:12.2f}{int8_ppl:14.3f}")
    print(f"\nsize: {fp32_mb/int8_mb:.2f}x smaller | "
          f"perplexity change: {(int8_ppl/fp32_ppl - 1)*100:+.1f}%")


if __name__ == "__main__":
    main()
