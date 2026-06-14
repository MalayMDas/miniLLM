"""GSM8K — grade-school math, scored by exact-match on the final number.

Generative (not multiple-choice): we few-shot prompt with chain-of-thought, let the
model generate, extract the last number it produced, and compare to the gold answer
(the value after '####'). No API, no judge — pure string match.

Honest note: a ~300M model scores near 0 here; the value is tracking the *delta*
from CoT/GRPO, and exercising the generative-eval path.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..scoring import continuation_logprob  # noqa: F401  (kept for symmetry)

# A couple of CoT exemplars so even a weak model emits the right format.
FEWSHOT = (
    "Question: Natalia sold 48 clips in April and half as many in May. How many "
    "clips did she sell altogether?\n"
    "Answer: In May she sold 48 / 2 = 24 clips. Altogether 48 + 24 = 72. #### 72\n\n"
    "Question: Weng earns $12 per hour. For 50 minutes she earned how much?\n"
    "Answer: Per minute she earns 12 / 60 = 0.2 dollars. For 50 minutes: 50 * 0.2 = "
    "10 dollars. #### 10\n\n"
)

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_answer(answer_field: str) -> str:
    """Gold GSM8K answer is the text after '####'."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def extract_pred(text: str) -> Optional[str]:
    """The model's answer = the last number it generated (after '####' if present)."""
    tail = text.split("####")[-1] if "####" in text else text
    nums = _NUM.findall(tail)
    return nums[-1].replace(",", "") if nums else None


def load_gsm8k(split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [{"question": ex["question"], "gold": gold_answer(ex["answer"])} for ex in ds]


def evaluate_gsm8k(model, tokenizer, examples: List[Dict], device: str = "cpu",
                   max_new_tokens: int = 200) -> float:
    from ...serve.generate import generate
    correct = 0
    for ex in examples:
        prompt = FEWSHOT + f"Question: {ex['question']}\nAnswer:"
        ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
        out = generate(model, ids, max_new_tokens=max_new_tokens, temperature=0.0,
                       top_k=None, top_p=None, stop_ids=[tokenizer.eos_id], device=device)
        pred = extract_pred(tokenizer.decode(out))
        correct += int(pred is not None and pred == ex["gold"])
    return correct / max(len(examples), 1)
