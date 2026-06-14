"""Byte-level BPE tokenizer (the default) — UTF-8 bytes + learned merges.

This is "from scratch" in the sense that we TRAIN our own merges on our own data
with our own special tokens; it is not a downloaded vocab. Because it is
byte-level, it still never hits an out-of-vocabulary token (same universality as
the raw-UTF-8 tokenizer) but compresses English ~4x, which is the whole point for
a tight compute budget.

Backed by the HuggingFace `tokenizers` library (fast Rust BPE).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from .byte_tokenizer import SPECIAL_TOKENS


class BPETokenizer:
    """Thin wrapper over a trained byte-level BPE `Tokenizer`."""

    def __init__(self, tk: Tokenizer) -> None:
        self.tk = tk

    # ---- training -------------------------------------------------------
    @classmethod
    def train(
        cls,
        corpus_iter: Iterable[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        tk = Tokenizer(models.BPE(unk_token=None))
        # Byte-level pre-tokenizer => operates on raw bytes; no UNK ever.
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tk.train_from_iterator(corpus_iter, trainer=trainer)
        return cls(tk)

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        self.tk.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # ---- properties -----------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def _sid(self, tok: str) -> int:
        return self.tk.token_to_id(tok)

    @property
    def pad_id(self) -> int:
        return self._sid("<|pad|>")

    @property
    def bos_id(self) -> int:
        return self._sid("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self._sid("<|eos|>")

    def token_to_id(self, tok: str) -> int:
        """Id of a (special) token. Used for chat/tool formatting."""
        return self._sid(tok)

    # ---- api ------------------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tk.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=False)
