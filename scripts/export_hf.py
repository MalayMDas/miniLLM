"""Export a checkpoint to a HuggingFace Llama folder (for TRL / vLLM / lm-eval / GGUF).

    python scripts/export_hf.py --ckpt artifacts/ckpt_sft_local/step_X.pt \
        --tokenizer artifacts/tok_local.json --out artifacts/hf_model

The result is a genuine `LlamaForCausalLM` (verified numerically identical to our
Decoder), so:
    - serve with vLLM:   vllm serve artifacts/hf_model
    - evaluate:          lm_eval --model hf --model_args pretrained=artifacts/hf_model
    - convert to GGUF:   scripts/export_gguf.py (uses llama.cpp)
Requires `transformers`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.model.hf_export import export_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tok_local.json")
    ap.add_argument("--out", default="artifacts/hf_model")
    args = ap.parse_args()
    export_hf(args.ckpt, args.out, tokenizer=args.tokenizer)


if __name__ == "__main__":
    main()
