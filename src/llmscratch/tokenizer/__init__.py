"""Tokenizer module: pick `byte` (UTF-8) or `bpe` (byte-level BPE) via config.

    from llmscratch.tokenizer import build_tokenizer
    tok = build_tokenizer({"mode": "bpe", "path": "artifacts/tok.json"})
"""
from __future__ import annotations

from typing import Protocol, List

from .byte_tokenizer import ByteTokenizer, SPECIAL_TOKENS
from .bpe import BPETokenizer


class Tokenizer(Protocol):
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    def encode(self, text: str, add_bos: bool = ..., add_eos: bool = ...) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...


def build_tokenizer(cfg: dict):
    """cfg keys: mode ('byte'|'bpe'), path (for bpe load)."""
    mode = cfg.get("mode", "bpe")
    if mode == "byte":
        return ByteTokenizer()
    if mode == "bpe":
        path = cfg.get("path")
        if not path:
            raise ValueError("bpe mode requires cfg['path'] to a trained tokenizer.json")
        return BPETokenizer.load(path)
    raise ValueError(f"unknown tokenizer mode: {mode}")


__all__ = ["ByteTokenizer", "BPETokenizer", "build_tokenizer", "SPECIAL_TOKENS"]
