"""Train the byte-level BPE tokenizer on a local corpus.

    python scripts/train_tokenizer.py --config configs/model_tiny.yaml

For 'byte' mode there is nothing to train — this script is a no-op and just
reminds you. For 'bpe' mode it trains merges and writes tokenizer.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# allow running from repo root without installing the package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.data import iter_local_lines
from llmscratch.tokenizer import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_tiny.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    tcfg = cfg["tokenizer"]
    if tcfg["mode"] == "byte":
        print("mode=byte -> no training needed (UTF-8 byte tokenizer is fixed).")
        return

    corpus = cfg["data"]["corpus"]
    out = Path(tcfg["path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"training byte-level BPE (vocab={tcfg['vocab_size']}) on {corpus} ...")
    tok = BPETokenizer.train(
        iter_local_lines(corpus),
        vocab_size=tcfg["vocab_size"],
        min_frequency=1,
    )
    tok.save(out)
    print(f"saved -> {out}  (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()
