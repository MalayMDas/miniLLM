"""Text data utilities for local smoke-training and (optional) HF streaming.

Two entry points:
  - iter_local_lines(path)         : read a local .txt file line by line (for tokenizer
                                     training + smoke-train; no internet needed).
  - PackedDataset                  : tokenize + pack a corpus into fixed-length blocks
                                     for next-token prediction.

HF streaming (FineWeb-Edu etc.) is wired in a later pretraining stage; kept out of
the local path so the scaffold runs with zero downloads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import torch
from torch.utils.data import Dataset


def iter_local_lines(path: str | Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


class PackedDataset(Dataset):
    """Concatenate token ids and slice into (block_size+1) windows.

    Each item returns (x, y) where y is x shifted by one — the standard
    next-token-prediction setup.
    """

    def __init__(self, token_ids: List[int], block_size: int):
        self.block_size = block_size
        n_blocks = (len(token_ids) - 1) // block_size
        if n_blocks < 1:
            raise ValueError("corpus too small for even one block; add more text")
        usable = n_blocks * block_size + 1
        self.data = torch.tensor(token_ids[:usable], dtype=torch.long)
        self.n_blocks = n_blocks

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int):
        s = i * self.block_size
        x = self.data[s : s + self.block_size]
        y = self.data[s + 1 : s + 1 + self.block_size]
        return x, y


def encode_corpus(tokenizer, texts: Iterator[str]) -> List[int]:
    ids: List[int] = [tokenizer.bos_id]
    for t in texts:
        ids.extend(tokenizer.encode(t, add_eos=True))
    return ids
