"""ChatML formatting + loss masking for SFT / instruct / tool data.

Format (one shared template across all post-training stages):

    <|im_start|>system\n{system}<|im_end|>
    <|im_start|>user\n{user}<|im_end|>
    <|im_start|>assistant\n{assistant}<|im_end|>

Key SFT detail: we train the model to produce ONLY the assistant turns, so prompt
tokens (system/user + the assistant header) are masked to -100 in the labels and
contribute no loss. We build ids from special-token IDs directly (not by encoding
the literal "<|im_start|>" string) so it works identically for byte and BPE
tokenizers regardless of how they pre-tokenize.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

IGNORE = -100


def _encode_turn(tok, role: str, content: str) -> List[int]:
    ids = [tok.token_to_id("<|im_start|>")]
    ids += tok.encode(f"{role}\n{content}")
    ids += [tok.token_to_id("<|im_end|>")]
    return ids


def render_chat(tok, messages: List[Dict[str, str]],
                train_on_assistant_only: bool = True) -> Tuple[List[int], List[int]]:
    """messages: [{role, content}, ...] -> (input_ids, labels).

    labels == input_ids on assistant tokens, IGNORE elsewhere (when masking on).
    """
    input_ids: List[int] = [tok.bos_id]
    labels: List[int] = [IGNORE]
    for msg in messages:
        turn = _encode_turn(tok, msg["role"], msg["content"])
        input_ids.extend(turn)
        if msg["role"] == "assistant" or not train_on_assistant_only:
            labels.extend(turn)               # learn to produce this turn
        else:
            labels.extend([IGNORE] * len(turn))
    input_ids.append(tok.eos_id)
    labels.append(tok.eos_id if not train_on_assistant_only else tok.eos_id)
    return input_ids, labels


def build_prompt(tok, messages: List[Dict[str, str]]) -> List[int]:
    """Inference-time prompt: all turns + an open assistant header to complete."""
    ids: List[int] = [tok.bos_id]
    for msg in messages:
        ids.extend(_encode_turn(tok, msg["role"], msg["content"]))
    ids.append(tok.token_to_id("<|im_start|>"))
    ids.extend(tok.encode("assistant\n"))
    return ids
