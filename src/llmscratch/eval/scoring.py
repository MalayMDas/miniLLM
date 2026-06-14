"""Core scoring primitive: log-probability of a continuation given a prefix.

This single function underlies both perplexity and multiple-choice accuracy — and
is exactly what lm-evaluation-harness calls a "loglikelihood request". Building it
ourselves makes the eval mechanics transparent; for official benchmark numbers,
wire the same model into lm-eval-harness (see eval/README in PLAN).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


@torch.no_grad()
def continuation_logprob(model, prefix_ids: List[int], cont_ids: List[int],
                         device: str = "cpu") -> float:
    """Sum log p(cont | prefix) under the model (teacher-forced).

    If prefix+cont exceeds the model's context, left-truncate the PREFIX (keep the
    full continuation at the end) — standard harness behavior for long contexts.
    """
    max_len = model.cfg.max_seq_len
    if len(prefix_ids) + len(cont_ids) > max_len:
        cont_ids = cont_ids[-(max_len - 1):]                 # guard: cont alone too long
        prefix_ids = prefix_ids[-(max_len - len(cont_ids)):]
    ids = torch.tensor([prefix_ids + cont_ids], device=device)
    logits, _ = model(ids)
    logits = logits[0]                              # (T, vocab)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    # token at position i is predicted by logits at i-1
    start = len(prefix_ids)
    total = 0.0
    for i in range(start, start + len(cont_ids)):
        total += logprobs[i - 1, ids[0, i]].item()
    return total


@torch.no_grad()
def sequence_nll(model, ids: List[int], device: str = "cpu") -> tuple[float, int]:
    """Return (sum negative log-likelihood, num predicted tokens) for a sequence."""
    ids = ids[: model.cfg.max_seq_len]
    x = torch.tensor([ids], device=device)
    logits, _ = model(x)
    logprobs = F.log_softmax(logits[0].float(), dim=-1)
    nll, n = 0.0, 0
    for i in range(1, len(ids)):
        nll -= logprobs[i - 1, x[0, i]].item()
        n += 1
    return nll, n
