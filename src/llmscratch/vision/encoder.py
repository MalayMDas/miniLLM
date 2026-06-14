"""Vision encoder TOGGLE: from-scratch ViT vs pretrained SigLIP.

This is the honest answer to "vision encoder from scratch" at a small budget:
keep it a config switch so you can compare. Both expose the same interface
(`forward(pixels) -> (B, num_tokens, out_dim)`, plus `.out_dim` / `.num_tokens`),
so the rest of the multimodal stack doesn't care which is used.

    cfg = {"mode": "from_scratch", "image_size": 64, "patch_size": 8, ...}
    cfg = {"mode": "pretrained", "model": "google/siglip-base-patch16-224", "freeze": true}
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .vit import ViT, ViTConfig


class PretrainedSigLIP(nn.Module):
    """Wraps a HF SigLIP vision tower. Requires `transformers`. Typically frozen."""
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name).vision_model
        self.out_dim = self.model.config.hidden_size
        self.num_tokens = (self.model.config.image_size // self.model.config.patch_size) ** 2
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state


def build_vision_encoder(cfg: Dict) -> nn.Module:
    mode = cfg.get("mode", "from_scratch")
    if mode == "from_scratch":
        return ViT(ViTConfig(
            image_size=cfg.get("image_size", 64),
            patch_size=cfg.get("patch_size", 8),
            dim=cfg.get("dim", 192),
            depth=cfg.get("depth", 4),
            heads=cfg.get("heads", 3),
        ))
    if mode == "pretrained":
        return PretrainedSigLIP(cfg["model"], freeze=cfg.get("freeze", True))
    raise ValueError(f"unknown vision encoder mode: {mode}")
