"""Train the byte-level BPE tokenizer.

    python scripts/train_tokenizer.py --config configs/model_tiny.yaml      # local corpus
    python scripts/train_tokenizer.py --config configs/tokenizer_32k.yaml   # HF stream

For 'byte' mode there is nothing to train (UTF-8 byte tokenizer is fixed). For 'bpe'
mode it trains merges and writes tokenizer.json. Source is local file or an HF
streaming sample (set tokenizer.source: hf) — the real 32k tokenizer is trained on
a FineWeb-Edu sample.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import yaml

# allow running from repo root without installing the package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.data import iter_local_lines
from llmscratch.tokenizer import BPETokenizer


def hf_text_iter(dataset: str, name: str, text_field: str, n_docs: int) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset(dataset, name=name, split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n_docs:
            break
        yield ex[text_field]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_tiny.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    tcfg = cfg["tokenizer"]
    if tcfg["mode"] == "byte":
        print("mode=byte -> no training needed (UTF-8 byte tokenizer is fixed).")
        return

    out = Path(tcfg["path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    if tcfg.get("source") == "hf":
        n = tcfg.get("sample_docs", 200000)
        print(f"training byte-level BPE (vocab={tcfg['vocab_size']}) on "
              f"{n} docs from {tcfg['hf_dataset']} ...")
        corpus_iter = hf_text_iter(tcfg["hf_dataset"], tcfg.get("hf_name"),
                                   tcfg.get("text_field", "text"), n)
    else:
        corpus = cfg["data"]["corpus"]
        print(f"training byte-level BPE (vocab={tcfg['vocab_size']}) on {corpus} ...")
        corpus_iter = iter_local_lines(corpus)

    tok = BPETokenizer.train(corpus_iter, vocab_size=tcfg["vocab_size"],
                             min_frequency=tcfg.get("min_frequency", 2))
    tok.save(out)
    print(f"saved -> {out}  (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()
