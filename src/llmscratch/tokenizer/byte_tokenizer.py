"""Byte-level (UTF-8) tokenizer — the "no training" option.

This is the literal answer to "why not just use UTF-8?". Vocab = 256 raw byte
values + a handful of special tokens. It never has out-of-vocabulary issues and
needs zero training, but produces ~4x longer sequences than byte-level BPE for
English text (see tokenizer/compare.py to measure it yourself).

Special tokens are placed ABOVE the 256 byte range so byte ids stay 0..255.
"""
from __future__ import annotations

from typing import List, Dict

# Shared special tokens so every stage (chat, tools, vision) agrees on ids.
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|im_start|>",   # chat turn start
    "<|im_end|>",     # chat turn end
    "<tool_call>",
    "</tool_call>",
    "<image>",        # multimodal placeholder
]


class ByteTokenizer:
    """UTF-8 byte tokenizer. id = byte value (0..255); specials start at 256."""

    def __init__(self) -> None:
        self.byte_vocab_size = 256
        self.special_to_id: Dict[str, int] = {
            tok: self.byte_vocab_size + i for i, tok in enumerate(SPECIAL_TOKENS)
        }
        self.id_to_special: Dict[int, str] = {v: k for k, v in self.special_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return self.byte_vocab_size + len(SPECIAL_TOKENS)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    def token_to_id(self, tok: str) -> int:
        """Id of a special token (e.g. '<|im_start|>'). Used for chat/tool formatting."""
        return self.special_to_id[tok]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Collect contiguous byte ids and decode; render specials inline.
        out, buf = [], bytearray()
        for i in ids:
            if i < self.byte_vocab_size:
                buf.append(i)
            else:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(self.id_to_special.get(i, ""))
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)
