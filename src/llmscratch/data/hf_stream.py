"""Streaming pretraining data from HuggingFace (e.g. FineWeb-Edu).

Streaming (not download-all) matters because pretraining corpora are TB-scale —
you can't fit them on disk. We pull examples lazily, tokenize on the fly, and pack
them into fixed-length blocks for next-token prediction.

    ds = packed_hf_stream(tokenizer, "HuggingFaceFW/fineweb-edu",
                          name="sample-10BT", block_size=2048)
    for x, y in DataLoader(ds, batch_size=...): ...

Requires `datasets`. Kept separate from the local-file path so the smoke-train
needs zero network access.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset


class PackedHFStream(IterableDataset):
    def __init__(self, tokenizer, dataset_name: str, block_size: int,
                 split: str = "train", name: Optional[str] = None,
                 text_field: str = "text", seed: int = 0,
                 buffer_docs: int = 1000):
        self.tok = tokenizer
        self.dataset_name = dataset_name
        self.name = name
        self.split = split
        self.text_field = text_field
        self.block_size = block_size
        self.seed = seed
        self.buffer_docs = buffer_docs

    def _doc_iter(self) -> Iterator[str]:
        from datasets import load_dataset
        ds = load_dataset(self.dataset_name, name=self.name, split=self.split,
                          streaming=True)
        ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_docs)
        # shard across DataLoader workers so they don't emit duplicate data
        info = torch.utils.data.get_worker_info()
        for i, ex in enumerate(ds):
            if info is None or (i % info.num_workers) == info.id:
                yield ex[self.text_field]

    def __iter__(self):
        buf: List[int] = []
        bs = self.block_size
        for text in self._doc_iter():
            buf.append(self.tok.bos_id)
            buf.extend(self.tok.encode(text, add_eos=True))
            while len(buf) >= bs + 1:
                chunk = buf[: bs + 1]
                buf = buf[bs:]                      # keep 1-token overlap for the shift
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def packed_hf_stream(tokenizer, dataset_name: str, block_size: int, **kw) -> PackedHFStream:
    return PackedHFStream(tokenizer, dataset_name, block_size, **kw)
