"""Measure the byte-vs-BPE tradeoff yourself.

    python -m llmscratch.tokenizer.compare --bpe artifacts/tok.json

Prints tokens/char and how much text fits in a fixed context for each scheme,
so the ~4x sequence-length difference is concrete rather than asserted.
"""
from __future__ import annotations

import argparse

from .byte_tokenizer import ByteTokenizer
from .bpe import BPETokenizer

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "Large language models learn to predict the next token. "
    "def add(a, b):\n    return a + b\n"
    "Café costs €3.50 — naïve façade. 日本語のテキストも。"
)


def stats(name: str, ids, text: str, ctx: int = 2048) -> None:
    n = len(ids)
    cpt = len(text) / max(n, 1)
    print(f"{name:>14}: {n:5d} tokens | {cpt:4.2f} chars/token | "
          f"~{int(cpt * ctx):6d} chars fit in ctx={ctx}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe", help="path to a trained byte-level BPE tokenizer.json")
    ap.add_argument("--text", default=SAMPLE)
    args = ap.parse_args()

    bt = ByteTokenizer()
    stats("UTF-8 bytes", bt.encode(args.text), args.text)

    if args.bpe:
        bp = BPETokenizer.load(args.bpe)
        stats("byte-BPE", bp.encode(args.text), args.text)
    else:
        print("(pass --bpe <tokenizer.json> to compare against trained BPE)")


if __name__ == "__main__":
    main()
