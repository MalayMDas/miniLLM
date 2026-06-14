"""Train the multimodal model (LLaVA-style) on the synthetic image->caption task.

    python scripts/train_vision.py --phase 1     # projector-only (alignment)
    python scripts/train_vision.py --phase 2     # train projector + LLM

Phase 1 freezes the vision encoder and the LLM and trains ONLY the projector — the
cheap alignment step. Phase 2 unfreezes the LLM for instruction tuning. The toy task
is learnable, so loss should fall and the model should start captioning colors.

Vision encoder is the config toggle (from-scratch ViT here; flip to SigLIP via
vision/encoder.py for real capability).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import ByteTokenizer
from llmscratch.vision import (build_vision_encoder, MultimodalDecoder,
                               SyntheticVLMDataset, make_vlm_collate)
from llmscratch.utils import build_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=2, choices=[1, 2])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--image-size", type=int, default=16)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tok = ByteTokenizer()                 # byte tokenizer keeps the demo dependency-free
    dim = 128
    decoder = Decoder(ModelConfig(vocab_size=tok.vocab_size, dim=dim, n_layers=4,
                                  n_heads=4, n_kv_heads=2, max_seq_len=128))
    encoder = build_vision_encoder({"mode": "from_scratch", "image_size": args.image_size,
                                    "patch_size": 8, "dim": dim, "depth": 3, "heads": 4})
    mm = MultimodalDecoder(decoder, encoder,
                           image_token_id=tok.token_to_id("<image>")).to(device)
    if args.phase == 1:
        mm.freeze_for_alignment()
        print("phase 1: training projector only")
    else:
        print("phase 2: training projector + LLM")

    ds = SyntheticVLMDataset(tok, tok.token_to_id("<image>"), encoder.num_tokens,
                             image_size=args.image_size)
    loader = DataLoader(ds, batch_size=6, shuffle=True,
                        collate_fn=make_vlm_collate(tok.pad_id))

    params = [p for p in mm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0)
    logger = build_logger({"backend": "tensorboard", "logdir": "runs"},
                          f"vision_phase{args.phase}", {})

    mm.train()
    step, it = 0, iter(loader)
    while step < args.steps:
        try:
            x, y, img = next(it)
        except StopIteration:
            it = iter(loader); x, y, img = next(it)
        x, y, img = x.to(device), y.to(device), img.to(device)
        _, loss = mm(x, img, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 30 == 0 or step == args.steps - 1:
            logger.log_scalar("train/loss", loss.item(), step)
            print(f"step {step:4d} | loss {loss.item():.4f}")
        step += 1
    logger.close()
    print("done.")


if __name__ == "__main__":
    main()
