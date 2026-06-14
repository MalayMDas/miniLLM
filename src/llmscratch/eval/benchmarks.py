"""Lightweight benchmarks: perplexity + multiple-choice accuracy.

Multiple-choice is scored the standard way — pick the option with the highest
length-normalized log-likelihood given the question (this is how HellaSwag / ARC /
MMLU are evaluated under lm-eval-harness's `acc_norm`).
"""
from __future__ import annotations

import math
from typing import Dict, List

from .scoring import continuation_logprob, sequence_nll


def perplexity(model, tokenizer, texts: List[str], device: str = "cpu") -> float:
    total_nll, total_tok = 0.0, 0
    for t in texts:
        ids = [tokenizer.bos_id] + tokenizer.encode(t)
        nll, n = sequence_nll(model, ids, device)
        total_nll += nll
        total_tok += n
    return math.exp(total_nll / max(total_tok, 1))


def multiple_choice_accuracy(model, tokenizer, examples: List[Dict], device: str = "cpu",
                             length_normalized: bool = True) -> float:
    """examples: [{"question": str, "choices": [str, ...], "answer": int}].

    answer is the index of the correct choice.
    """
    correct = 0
    for ex in examples:
        prefix = [tokenizer.bos_id] + tokenizer.encode(ex["question"])
        scores = []
        for choice in ex["choices"]:
            cont = tokenizer.encode(" " + choice)
            lp = continuation_logprob(model, prefix, cont, device)
            scores.append(lp / len(cont) if length_normalized and cont else lp)
        if max(range(len(scores)), key=lambda i: scores[i]) == ex["answer"]:
            correct += 1
    return correct / max(len(examples), 1)
