"""In-memory vector store with cosine top-k search.

A teaching stand-in for FAISS / Chroma / pgvector. The interface (add / search)
matches what those provide, so the pipeline doesn't change when you scale up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Doc:
    text: str
    meta: dict = field(default_factory=dict)


class VectorStore:
    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.docs: List[Doc] = []

    def add(self, vectors: np.ndarray, docs: List[Doc]) -> None:
        self.docs.extend(docs)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[tuple[Doc, float]]:
        if self.vectors is None or len(self.docs) == 0:
            return []
        sims = self.vectors @ query_vec            # vectors are L2-normalized => cosine
        k = min(k, len(self.docs))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.docs[i], float(sims[i])) for i in top]
