"""RAG pipeline: chunk -> embed -> store -> retrieve -> grounded generation.

Demonstrates the core RAG ideas: retrieval grounds the answer in real documents
and lets you cite sources, which mitigates hallucination. Generation is optional —
pass a model+tokenizer to get an answer, or use `.retrieve()` alone to inspect what
context would be fed in.
"""
from __future__ import annotations

import re
from typing import List, Optional

from .embedder import HashingEmbedder
from .store import Doc, VectorStore


def chunk_text(text: str, max_words: int = 40) -> List[str]:
    """Naive sentence-aware chunking (production: token-based with overlap)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], []
    for s in sentences:
        cur.append(s)
        if sum(len(c.split()) for c in cur) >= max_words:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class RAGPipeline:
    def __init__(self, embedder: Optional[HashingEmbedder] = None):
        self.embedder = embedder or HashingEmbedder()
        self.store = VectorStore()

    def add_documents(self, docs: List[str], source: str = "doc") -> int:
        chunks: List[str] = []
        metas: List[dict] = []
        for di, d in enumerate(docs):
            for ci, ch in enumerate(chunk_text(d)):
                chunks.append(ch)
                metas.append({"source": f"{source}[{di}]", "chunk": ci})
        vecs = self.embedder.embed_batch(chunks)
        self.store.add(vecs, [Doc(text=c, meta=m) for c, m in zip(chunks, metas)])
        return len(chunks)

    def retrieve(self, question: str, k: int = 3):
        return self.store.search(self.embedder.embed(question), k)

    def build_prompt(self, question: str, k: int = 3):
        hits = self.retrieve(question, k)
        context = "\n".join(f"[{i+1}] {doc.text}" for i, (doc, _) in enumerate(hits))
        user = (f"Use the context to answer. Cite sources like [1].\n\n"
                f"Context:\n{context}\n\nQuestion: {question}")
        return [{"role": "user", "content": user}], hits

    def answer(self, model, tokenizer, question: str, k: int = 3,
               device: str = "cpu", max_new_tokens: int = 128):
        from ...serve.generate import generate_chat
        messages, hits = self.build_prompt(question, k)
        text = generate_chat(model, tokenizer, messages,
                             max_new_tokens=max_new_tokens, device=device)
        return {"answer": text, "sources": [d.meta for d, _ in hits]}
