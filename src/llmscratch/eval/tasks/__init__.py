"""Benchmark task loaders + scorers (all local; one-time HF dataset download)."""
from .mcq_tasks import load_hellaswag, load_openbookqa
from .gsm8k import load_gsm8k, evaluate_gsm8k, extract_pred, gold_answer
from .bfcl import load_bfcl_jsonl, evaluate_bfcl, call_matches
from .vqa import vqa_accuracy, normalize_answer, evaluate_vqa

__all__ = [
    "load_hellaswag", "load_openbookqa",
    "load_gsm8k", "evaluate_gsm8k", "extract_pred", "gold_answer",
    "load_bfcl_jsonl", "evaluate_bfcl", "call_matches",
    "vqa_accuracy", "normalize_answer", "evaluate_vqa",
]
