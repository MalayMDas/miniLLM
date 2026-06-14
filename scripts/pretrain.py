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
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.data import iter_local_lines, encode_corpus, PackedDataset
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import build_tokenizer
from llmscratch.train import Trainer, TrainArgs
from llmscratch.utils import (load_config, build_logger, run_id,
                              find_latest, load_checkpoint, setup_distributed, cleanup)


def build_loader(cfg, tok, dist):
    d = cfg["data"]
    bs = cfg["train"]["batch_size"]
    if d["source"] == "local":
        ids = encode_corpus(tok, iter_local_lines(d["corpus"]))
        ds = PackedDataset(ids, d["block_size"])
        sampler = (DistributedSampler(ds, num_replicas=dist.world_size, rank=dist.rank,
                                      drop_last=True) if dist.distributed else None)
        return DataLoader(ds, batch_size=bs, shuffle=(sampler is None),
                          sampler=sampler, drop_last=True)
    elif d["source"] == "hf":
        from llmscratch.data.hf_stream import packed_hf_stream
        ds = packed_hf_stream(tok, d["hf_dataset"], d["block_size"],
                              name=d.get("hf_name"), text_field=d.get("text_field", "text"),
                              rank=dist.rank, world_size=dist.world_size)
        return DataLoader(ds, batch_size=bs)   # IterableDataset: shards itself by rank/worker
    raise ValueError(f"unknown data.source: {d['source']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_tiny.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dist = setup_distributed()
    torch.manual_seed(cfg["train"]["seed"] + dist.rank)
    device = dist.device

    tok = build_tokenizer(cfg["tokenizer"])
    m = cfg["model"]
    model = Decoder(ModelConfig(
        vocab_size=tok.vocab_size, dim=m["dim"], n_layers=m["n_layers"],
        n_heads=m["n_heads"], n_kv_heads=m["n_kv_heads"],
        max_seq_len=m["max_seq_len"], dropout=m["dropout"])).to(device)
    if dist.is_main:
        print(f"params: {model.num_params()/1e6:.2f}M | device: {device} | "
              f"world_size: {dist.world_size}")

    if dist.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_ids = [dist.local_rank] if device.startswith("cuda") else None
        model = DDP(model, device_ids=ddp_ids)

    loader = build_loader(cfg, tok, dist)
    t = cfg["train"]
    targs = TrainArgs(
        steps=t["steps"], lr=t["lr"], warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"], grad_accum=t["grad_accum"],
        device=device, amp=t["amp"], log_every=t["log_every"],
        ckpt_every=t["ckpt_every"], ckpt_dir=t["ckpt_dir"], is_main=dist.is_main)

    logger = build_logger(cfg.get("logging", {}) if dist.is_main else {"backend": "none"},
                          run_id(cfg, "pretrain"), cfg)
    if dist.is_main:
        logger.log_hparams(cfg)
    trainer = Trainer(model, loader, targs, logger=logger,
                      tokens_per_step=t["batch_size"] * cfg["data"]["block_size"] * dist.world_size)

    start = 0
    latest = find_latest(t["ckpt_dir"])
    if latest is not None:
        start = load_checkpoint(latest, trainer.raw_model, trainer.opt, map_location=device)
        if dist.is_main:
            print(f"resumed from {latest} at step {start}")

    trainer.train(start_step=start)
    cleanup()
    if dist.is_main:
        print("done.")


if __name__ == "__main__":
    main()
