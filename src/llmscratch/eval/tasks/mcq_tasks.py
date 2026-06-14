"""HellaSwag + OpenBookQA — multiple-choice, scored by log-likelihood.

Both reduce to our normalized MCQ form ({question, choices, answer}) and are scored
with eval.benchmarks.multiple_choice_accuracy (length-normalized log-likelihood,
i.e. lm-eval-harness `acc_norm`). Datasets load from the HF Hub (one-time download,
no API). Pass `limit` to evaluate a subset for a quick local check.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def load_hellaswag(split: str = "validation", limit: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for ex in ds:
        ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).strip() if ex.get("ctx_b") else ex["ctx"]
        out.append({"question": ctx, "choices": ex["endings"], "answer": int(ex["label"])})
    return out


def load_openbookqa(split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("openbookqa", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for ex in ds:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        answer = labels.index(ex["answerKey"])
        out.append({"question": ex["question_stem"], "choices": texts, "answer": answer})
    return out
