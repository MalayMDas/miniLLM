"""LLaVA-style multimodal model: vision encoder + projector + text decoder.

Mechanic: the prompt reserves `num_tokens` <image> placeholder tokens per image.
We embed the text normally, run the image through the encoder+projector to get the
same number of vision embeddings, and splice them into the <image> positions before
running the decoder. The decoder then attends over text and image tokens uniformly.

Training (LLaVA recipe):
  phase 1 — freeze encoder + LLM, train only the projector (alignment).
  phase 2 — unfreeze LLM (and optionally encoder), train on multimodal instructions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .projector import Projector


class MultimodalDecoder(nn.Module):
    def __init__(self, decoder, vision_encoder, image_token_id: int,
                 projector_hidden: int | None = None):
        super().__init__()
        self.decoder = decoder
        self.vision = vision_encoder
        self.projector = Projector(vision_encoder.out_dim, decoder.cfg.dim, projector_hidden)
        self.image_token_id = image_token_id
        self.num_image_tokens = vision_encoder.num_tokens

    def freeze_for_alignment(self) -> None:
        """Phase 1: only the projector trains."""
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.vision.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                targets: torch.Tensor | None = None):
        embeds = self.decoder.tok_emb(input_ids)              # (B, T, dim)
        vis = self.projector(self.vision(pixel_values))       # (B, num_tokens, dim)

        mask = input_ids == self.image_token_id               # (B, T)
        n_slots = int(mask.sum().item())
        expected = vis.shape[0] * vis.shape[1]
        if n_slots != expected:
            raise ValueError(f"found {n_slots} <image> tokens but vision produced "
                             f"{expected} embeddings — each image needs exactly "
                             f"{self.num_image_tokens} placeholder tokens")
        # row-major scatter: image tokens filled in order with the projected patches
        embeds = embeds.clone()
        embeds[mask] = vis.reshape(-1, vis.shape[-1]).to(embeds.dtype)

        return self.decoder(inputs_embeds=embeds, targets=targets)
