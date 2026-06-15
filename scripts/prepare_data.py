"""Pre-download + tokenize a fixed sample to a local .bin (so training is offline).

    python scripts/prepare_data.py --tokenizer artifacts/tok_local.json --tokens 100000000

Streams the dataset ONCE, tokenizes each document with our BPE, and appends the
token ids to data/<out>.bin as uint16. After this, point pretrain at it
(`data.source: bin`) and training does NO network I/O — no streaming timeouts.

Tip: `pip install hf_xet` first for much faster/steadier downloads (the HF Xet CDN).
Resumable-ish: if interrupted, just re-run with a smaller --tokens or delete the
partial .bin. Needs vocab_size <= 65535 (uint16).
"""
from __future__ import annotations

# datasets first (pyarrow before anything heavy); this script doesn't import torch.
try:
    import datasets  # noqa: F401
except Exception:
    pass

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.tokenizer import build_tokenizer


def _norm_name(name):
    """Treat 'none'/'null'/'' as None so config-less datasets (MiniPile) load."""
    if name is None or str(name).strip().lower() in ("", "none", "null"):
        return None
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="artifacts/tok_local.json")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT",
                    help="dataset config name; pass 'none' (or '') for datasets that "
                         "have no config, e.g. MiniPile")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--tokens", type=int, default=100_000_000, help="target token count")
    ap.add_argument("--out", default="data/fineweb_local.bin")
    # MIX multiple datasets (e.g. web text + code). Comma-separated, equal length.
    ap.add_argument("--datasets", default=None,
                    help="comma list to MIX, e.g. 'HuggingFaceFW/fineweb-edu,bigcode/the-stack-smol'")
    ap.add_argument("--names", default=None, help="comma list of config names (use 'none')")
    ap.add_argument("--weights", default=None, help="comma list of mixing weights")
    ap.add_argument("--text-fields", default=None, help="comma list of text fields (code uses 'content')")
    ap.add_argument("--data-dir", default=None, help="single-dataset data_dir (subset)")
    ap.add_argument("--data-dirs", default=None,
                    help="comma list of data_dirs for --datasets (blank entry = none); "
                         "e.g. the-stack-smol code lives under 'data/python'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = build_tokenizer({"mode": "bpe", "path": args.tokenizer})
    if tok.vocab_size > 65535:
        raise SystemExit(f"vocab_size {tok.vocab_size} > 65535 — use uint32 (edit dtype)")

    from datasets import load_dataset

    def doc_stream(dataset, name, field, data_dir=None):
        ds = load_dataset(dataset, name=_norm_name(name), split="train", streaming=True,
                          data_dir=(data_dir or None))
        for ex in ds:
            yield ex[field]

    # Build the (possibly mixed) document iterator.
    if args.datasets:
        dsets = [s.strip() for s in args.datasets.split(",")]
        n = len(dsets)
        names = (args.names.split(",") if args.names else ["none"] * n)
        fields = (args.text_fields.split(",") if args.text_fields else ["text"] * n)
        weights = ([float(w) for w in args.weights.split(",")] if args.weights else [1.0] * n)
        ddirs = (args.data_dirs.split(",") if args.data_dirs else [""] * n)
        if not (len(names) == len(fields) == len(weights) == len(ddirs) == n):
            raise SystemExit("--datasets/--names/--weights/--text-fields/--data-dirs must have equal length")
        from llmscratch.data.mixing import weighted_interleave
        print(f"mixing {n} datasets: " +
              ", ".join(f"{d}({w})" for d, w in zip(dsets, weights)))
        streams = [doc_stream(d, nm.strip(), fl.strip(), dd.strip())
                   for d, nm, fl, dd in zip(dsets, names, fields, ddirs)]
        docs = weighted_interleave(streams, weights, seed=args.seed)
        meta_src = {"datasets": dsets, "weights": weights}
    else:
        docs = doc_stream(args.dataset, args.name, args.text_field, args.data_dir)
        meta_src = {"dataset": args.dataset, "name": args.name}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    print(f"writing up to {args.tokens:,} tokens -> {out} (uint16) ...")
    with open(out, "wb") as f:
        for text in docs:
            ids = [tok.bos_id] + tok.encode(text, add_eos=True)
            np.asarray(ids, dtype=np.uint16).tofile(f)
            written += len(ids)
            if written % 2_000_000 < len(ids):
                print(f"  {written:,} / {args.tokens:,} tokens "
                      f"({100*written/args.tokens:.0f}%)")
            if written >= args.tokens:
                break

    meta = {"tokens": written, "vocab_size": tok.vocab_size, "dtype": "uint16", **meta_src}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {written:,} tokens -> {out} ({out.stat().st_size/1e6:.0f} MB). "
          f"Set data.source: bin, data.bin_path: {out} to train offline.")


if __name__ == "__main__":
    main()
