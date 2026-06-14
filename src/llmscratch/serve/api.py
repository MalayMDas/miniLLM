"""Minimal OpenAI-compatible chat API over our model (FastAPI).

    pip install fastapi uvicorn
    python -m llmscratch.serve.api --ckpt artifacts/ckpt_sft/step_0000150.pt \
        --tokenizer artifacts/tok.json

    curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'

This is for understanding the serving contract. For real throughput, serve the
HF-wrapped checkpoint with **vLLM** (`vllm serve <model>`), which gives the same
OpenAI API plus PagedAttention + continuous batching. We keep the interface
identical so a client doesn't care which is running.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..model import Decoder, ModelConfig
from ..tokenizer import build_tokenizer
from ..utils.checkpoint import load_checkpoint
from .generate import generate_chat


def build_app(model, tokenizer, device: str):
    from fastapi import FastAPI
    from pydantic import BaseModel

    class Msg(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: list[Msg]
        max_tokens: int = 128
        temperature: float = 0.8

    app = FastAPI(title="llmscratch")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        msgs = [m.model_dump() for m in req.messages]
        text = generate_chat(model, tokenizer, msgs, max_new_tokens=req.max_tokens,
                             temperature=req.temperature, device=device)
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}]}

    return app


def _load(ckpt_path: str, tokenizer_path: str, device: str):
    tok = build_tokenizer({"mode": "bpe", "path": tokenizer_path})
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["model_config"])
    model = Decoder(cfg)
    load_checkpoint(ckpt_path, model, map_location=device)
    return model.to(device).eval(), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tok.json")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = _load(args.ckpt, args.tokenizer, device)
    import uvicorn
    uvicorn.run(build_app(model, tok, device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
