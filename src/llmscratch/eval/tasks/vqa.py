"""VQAv2 — visual question answering, scored with the official VQA soft-accuracy.

VQA accuracy is local string matching (no API): each question has 10 human answers,
and acc = min(#humans-who-said-the-pred / 3, 1), averaged over questions. We apply
the standard answer normalization (lowercase, strip punctuation/articles, number
words). The real dataset needs COCO images (large download); the metric + runner
here are exact and unit-tested on fixtures.
"""
from __future__ import annotations

import re
from typing import Dict, List

_ARTICLES = {"a", "an", "the"}
_NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_PUNCT = re.compile(r"[^\w\s]")


def normalize_answer(ans: str) -> str:
    ans = ans.lower().strip()
    ans = _PUNCT.sub("", ans)
    toks = [_NUMBERS.get(t, t) for t in ans.split() if t not in _ARTICLES]
    return " ".join(toks)


def vqa_accuracy(pred: str, human_answers: List[str]) -> float:
    """Official metric: min(matches / 3, 1)."""
    p = normalize_answer(pred)
    matches = sum(1 for a in human_answers if normalize_answer(a) == p)
    return min(matches / 3.0, 1.0)


def evaluate_vqa(mm_model, tokenizer, examples: List[Dict], device: str = "cpu",
                 max_new_tokens: int = 16) -> float:
    """examples: [{image: Tensor(3,H,W), question: str, answers: [str]*10}].

    Requires a trained MultimodalDecoder. Generation here is a simple greedy decode
    over the spliced image+text prompt (multimodal generate is intentionally minimal).
    """
    import torch
    from ...data.chat import IGNORE  # noqa: F401
    total = 0.0
    img_id = mm_model.image_token_id
    for ex in examples:
        ids = ([tokenizer.bos_id] + [img_id] * mm_model.num_image_tokens
               + tokenizer.encode(ex["question"]))
        x = torch.tensor([ids], device=device)
        pix = ex["image"].unsqueeze(0).to(device)
        gen = []
        for _ in range(max_new_tokens):
            logits, _ = mm_model(x, pix)
            nxt = int(logits[0, -1].argmax())
            if nxt == tokenizer.eos_id:
                break
            gen.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        total += vqa_accuracy(tokenizer.decode(gen), ex["answers"])
    return total / max(len(examples), 1)
