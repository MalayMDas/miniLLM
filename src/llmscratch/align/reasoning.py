"""Reasoning stage helpers.

Two complementary paths (see PLAN.md §Stage 6):

  (a) CoT distillation — the cheap, effective one at small scale. You SFT on
      chain-of-thought traces (from a strong teacher), formatted with explicit
      <think>...</think> reasoning before the final answer. This reuses the SFT
      machinery; this module just provides the formatting.

  (b) RL with verifiable rewards (GRPO) — see align/grpo.py.

A reasoning trace is rendered as a normal assistant turn whose content is
"<think>{reasoning}</think>{answer}", so loss masking / templating are unchanged.
"""
from __future__ import annotations

from typing import Dict, List


def format_cot_turn(reasoning: str, answer: str) -> str:
    return f"<think>{reasoning}</think>{answer}"


def cot_conversation(question: str, reasoning: str, answer: str,
                     system: str | None = None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    msgs.append({"role": "assistant", "content": format_cot_turn(reasoning, answer)})
    return msgs


def extract_answer(text: str) -> str:
    """Strip the <think>...</think> block, returning the final answer only."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()
