"""Train from a pre-tokenized local .bin (no network during training).

`scripts/prepare_data.py` streams a fixed sample once and writes all token ids to a
flat uint16 file. This dataset memory-maps that file and yields random fixed-length
windows (the nanoGPT approach): every step draws a random offset, so the whole local
corpus is covered and resume just continues sampling — no data-position bookkeeping,
no mid-training downloads, no streaming timeouts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset


class BinDataset(IterableDataset):
    def __init__(self, bin_path: str | Path, block_size: int, seed: int = 0,
                 rank: int = 0, world_size: int = 1, dtype=np.uint16):
        self.bin_path = str(bin_path)
        self.block_size = block_size
        self.seed = seed
        self.rank = rank
        self.dtype = dtype

    def __iter__(self):
        data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        n = len(data)
        if n <= self.block_size + 1:
            raise ValueError(f"{self.bin_path} has only {n} tokens; need > block_size+1")
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        # decorrelate ranks/workers so they don't draw identical windows
        rng = np.random.default_rng(self.seed + 1000 * self.rank + wid)
        bs = self.block_size
        while True:                                    # infinite; Trainer bounds length
            i = int(rng.integers(0, n - bs - 1))
            chunk = np.asarray(data[i : i + bs + 1]).astype(np.int64)
            yield torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])
