"""Generation utilities for serving: stop tokens, nucleus sampling, chat helper.

Our Decoder.generate is fine for demos; this adds the serving niceties (top-p, stop
at <|im_end|>, return only the new text). For production throughput you'd serve the
HF-wrapped model with vLLM (PagedAttention, continuous batching) — see serve/api.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..data.chat import build_prompt


@torch.no_grad()
def generate(model, input_ids: List[int], max_new_tokens: int = 128,
             temperature: float = 0.8, top_k: Optional[int] = 40,
             top_p: Optional[float] = 0.95, stop_ids: Optional[List[int]] = None,
             device: str = "cpu") -> List[int]:
    model.eval()
    idx = torch.tensor([input_ids], device=device)
    stop = set(stop_ids or [])
    new: List[int] = []
    for _ in range(max_new_tokens):
        cond = idx[:, -model.cfg.max_seq_len:]
        logits, _ = model(cond)
        logits = logits[:, -1, :].float() / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        if top_p:
            logits = _top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        tok_id = int(nxt.item())
        if tok_id in stop:
            break
        new.append(tok_id)
        idx = torch.cat([idx, nxt], dim=1)
    return new


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    scatter_remove = remove.scatter(-1, sorted_idx, remove)
    return logits.masked_fill(scatter_remove, -float("inf"))


def generate_chat(model, tokenizer, messages: List[Dict[str, str]],
                  max_new_tokens: int = 128, device: str = "cpu", **kw) -> str:
    """Build a ChatML prompt, generate the assistant turn, return only its text."""
    prompt = build_prompt(tokenizer, messages)
    stop = [tokenizer.token_to_id("<|im_end|>"), tokenizer.eos_id]
    out = generate(model, prompt, max_new_tokens=max_new_tokens, stop_ids=stop,
                   device=device, **kw)
    return tokenizer.decode(out)
