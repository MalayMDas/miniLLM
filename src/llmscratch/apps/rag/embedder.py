"""Text embedders for retrieval.

Default is a dependency-free hashing bag-of-words embedder (deterministic via
hashlib) — enough to demonstrate retrieval offline. For real semantic search,
swap in `sentence-transformers` (or embeddings from our own trained model); the
`embed`/`embed_batch` interface stays the same, which is the point of the
abstraction.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _hash(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            vec[self._hash(tok)] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim))
