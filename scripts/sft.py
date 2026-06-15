"""Instruct SFT entrypoint.

    python scripts/sft.py --config configs/sft_tiny.yaml

Loads a base checkpoint (init_from), fine-tunes on chat data with assistant-only
loss masking, then prints a sample chat completion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.align import SFTDataset, make_collate, load_chat_jsonl
from llmscratch.data.chat import build_prompt
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import build_tokenizer
from llmscratch.train import Trainer, TrainArgs
from llmscratch.utils import (load_config, build_logger, run_id, pick_device,
                              load_checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_tiny.yaml")
    ap.add_argument("--init-from", default=None,
                    help="base checkpoint path (overrides config init_from)")
    ap.add_argument("--chat-jsonl", default=None,
                    help="override data.chat_jsonl; comma-separated paths are MERGED "
                         "(e.g. instruct + tools + safety)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="override train.ckpt_dir (separate output dir for this pass)")
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

    init_from = args.init_from or cfg.get("init_from")
    if init_from and Path(init_from).exists():
        load_checkpoint(init_from, model, map_location="cpu")
        print(f"initialized from base checkpoint: {init_from}")
    else:
        print("WARNING: no base checkpoint — SFT from random init (smoke test only)")

    chat_arg = args.chat_jsonl or cfg["data"]["chat_jsonl"]
    convs = []
    for pth in [p.strip() for p in chat_arg.split(",") if p.strip()]:
        n0 = len(convs)
        convs += load_chat_jsonl(pth)
        print(f"  loaded {len(convs)-n0} conversations from {pth}")
    print(f"SFT on {len(convs)} conversations total")
    ds = SFTDataset(tok, convs, max_len=cfg["data"]["max_len"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        collate_fn=make_collate(tok.pad_id))

    t = cfg["train"]
    targs = TrainArgs(steps=t["steps"], lr=t["lr"], warmup_steps=t["warmup_steps"],
                      weight_decay=t["weight_decay"], grad_accum=t["grad_accum"],
                      device=device, amp=t["amp"], log_every=t["log_every"],
                      ckpt_every=t["ckpt_every"], ckpt_dir=args.ckpt_dir or t["ckpt_dir"],
                      time_budget_min=t.get("time_budget_min"))
    logger = build_logger(cfg.get("logging", {}), run_id(cfg, "sft"), cfg)
    Trainer(model, loader, targs, logger=logger).train()

    # sample chat
    model.eval()
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    ids = torch.tensor([build_prompt(tok, msgs)], device=device)
    out = model.generate(ids, max_new_tokens=30, temperature=0.7, top_k=40)
    print("\n--- chat sample ---")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
