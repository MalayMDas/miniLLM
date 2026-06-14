"""A small Vision Transformer (ViT) trained from scratch.

Splits an image into patches, embeds each as a token, and runs a (non-causal)
transformer encoder. Output is one embedding per patch — exactly what a LLaVA-style
projector needs to feed into the LLM.

Honest note: a from-scratch ViT at this scale/budget is weak vs a pretrained
SigLIP/CLIP encoder (see vision/encoder.py for the toggle). It's here for the
learning value and so the multimodal plumbing is real end to end.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViTConfig:
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 3
    dim: int = 192
    depth: int = 4
    heads: int = 3
    mlp_ratio: float = 4.0

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class ViTBlock(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3, bias=True)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        self.norm2 = nn.LayerNorm(cfg.dim)
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, cfg.dim))

    def forward(self, x):
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)   # non-causal (bidirectional)
        attn = attn.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(cfg.in_channels, cfg.dim,
                                     kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(ViTBlock(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.dim)
        self.out_dim = cfg.dim
        self.num_tokens = cfg.num_patches

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B, C, H, W) -> (B, num_patches, dim)
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
