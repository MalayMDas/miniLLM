"""Base pretraining entrypoint (single-process; multi-GPU via torchrun later).

    python scripts/pretrain.py --config configs/pretrain_tiny.yaml

Resumes from the latest checkpoint in train.ckpt_dir if present (spot-safe).
Supports a local corpus (offline) or HF streaming (set data.source: hf).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.data import iter_local_lines, encode_corpus, PackedDataset
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import build_tokenizer
from llmscratch.train import Trainer, TrainArgs
from llmscratch.utils import (load_config, build_logger, run_id, pick_device,
                              find_latest, load_checkpoint)


def build_loader(cfg, tok):
    d = cfg["data"]
    bs = cfg["train"]["batch_size"]
    if d["source"] == "local":
        ids = encode_corpus(tok, iter_local_lines(d["corpus"]))
        ds = PackedDataset(ids, d["block_size"])
        return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    elif d["source"] == "hf":
        from llmscratch.data.hf_stream import packed_hf_stream
        ds = packed_hf_stream(tok, d["hf_dataset"], d["block_size"],
                              name=d.get("hf_name"), text_field=d.get("text_field", "text"))
        return DataLoader(ds, batch_size=bs)   # IterableDataset: no shuffle/sampler
    raise ValueError(f"unknown data.source: {d['source']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_tiny.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg["train"]["seed"])

    device = pick_device(cfg["train"]["device"])
    tok = build_tokenizer(cfg["tokenizer"])
    m = cfg["model"]
    model = Decoder(ModelConfig(
        vocab_size=tok.vocab_size, dim=m["dim"], n_layers=m["n_layers"],
        n_heads=m["n_heads"], n_kv_heads=m["n_kv_heads"],
        max_seq_len=m["max_seq_len"], dropout=m["dropout"]))
    print(f"params: {model.num_params()/1e6:.2f}M | device: {device}")

    loader = build_loader(cfg, tok)
    t = cfg["train"]
    targs = TrainArgs(
        steps=t["steps"], lr=t["lr"], warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"], grad_accum=t["grad_accum"],
        device=device, amp=t["amp"], log_every=t["log_every"],
        ckpt_every=t["ckpt_every"], ckpt_dir=t["ckpt_dir"])

    logger = build_logger(cfg.get("logging", {}), run_id(cfg, "pretrain"), cfg)
    logger.log_hparams(cfg)
    trainer = Trainer(model, loader, targs, logger=logger,
                      tokens_per_step=t["batch_size"] * cfg["data"]["block_size"])

    start = 0
    latest = find_latest(t["ckpt_dir"])
    if latest is not None:
        start = load_checkpoint(latest, model, trainer.opt, map_location=device)
        print(f"resumed from {latest} at step {start}")

    trainer.train(start_step=start)
    print("done.")


if __name__ == "__main__":
    main()
