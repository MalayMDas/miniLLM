"""Provide your own prompt to a trained checkpoint — interactive or single-shot.

    # one-shot:
    python scripts/chat.py --ckpt artifacts/ckpt_sft_local/step_0000400.pt \
        --tokenizer artifacts/tok_local.json --mode chat --prompt "Hello, who are you?"

    # interactive REPL (type prompts, Ctrl-C or 'exit' to quit):
    python scripts/chat.py --ckpt artifacts/ckpt_pretrain_local/step_XXXX.pt \
        --tokenizer artifacts/tok_local.json --mode complete

Modes:
  complete : raw text continuation (use for a BASE / pretrained checkpoint).
  chat     : ChatML instruct format (use for an SFT / instruct checkpoint).

Sampling knobs: --temperature, --top-k, --top-p, --max-new-tokens, --system.
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
from llmscratch.serve.generate import generate, generate_chat


def respond(model, tok, prompt, mode, device, args, system) -> str:
    if mode == "chat":
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        return generate_chat(model, tok, msgs, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             top_p=args.top_p, device=device)
    ids = [tok.bos_id] + tok.encode(prompt)
    out = generate(model, ids, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                   stop_ids=[tok.eos_id], device=device)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tok_local.json")
    ap.add_argument("--mode", choices=["chat", "complete"], default="complete")
    ap.add_argument("--prompt", default=None, help="single-shot prompt (omit for REPL)")
    ap.add_argument("--system", default=None, help="system prompt (chat mode)")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer({"mode": "bpe", "path": args.tokenizer})
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Decoder(ModelConfig(**payload["model_config"]))
    load_checkpoint(args.ckpt, model, map_location=device)
    model.to(device).eval()
    print(f"loaded {args.ckpt} ({model.num_params()/1e6:.1f}M params) | mode={args.mode} | device={device}")

    if args.prompt is not None:
        print(respond(model, tok, args.prompt, args.mode, device, args, args.system))
        return

    print("Interactive — type a prompt ('exit' or Ctrl-C to quit).")
    try:
        while True:
            prompt = input("\n>>> ").strip()
            if prompt.lower() in {"exit", "quit"}:
                break
            if prompt:
                print(respond(model, tok, prompt, args.mode, device, args, args.system))
    except (KeyboardInterrupt, EOFError):
        print("\nbye.")


if __name__ == "__main__":
    main()
