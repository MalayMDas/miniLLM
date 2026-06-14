"""Vision->LLM projector (the only thing trained in LLaVA phase 1).

A small MLP mapping vision-encoder embeddings into the LLM's hidden dim, so image
patches become "tokens" the decoder can attend to.
"""
from __future__ import annotations

import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or llm_dim
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x):
        return self.net(x)
