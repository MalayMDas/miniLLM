"""Synthetic image->caption data so the multimodal stack is trainable offline.

Each example is a solid-color image and the caption "a {color} image". It's a toy,
but it's genuinely learnable: the model must read the projected image tokens to
predict the color word — so a dropping loss proves vision actually informs the LLM.
Swap this for a real image-text dataset (COCO / LLaVA mixture) at scale; the
(input_ids, pixel_values, targets) contract is unchanged.
"""
from __future__ import annotations

from typing import List

import torch
from torch.utils.data import Dataset

from ..data.chat import IGNORE

COLORS = {
    "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0), "purple": (0.5, 0.0, 0.5), "white": (1.0, 1.0, 1.0),
}


class SyntheticVLMDataset(Dataset):
    def __init__(self, tokenizer, image_token_id: int, num_image_tokens: int,
                 image_size: int = 16, length: int = 240):
        self.tok = tokenizer
        self.image_token_id = image_token_id
        self.num_image_tokens = num_image_tokens
        self.image_size = image_size
        self.length = length
        self.names = list(COLORS.keys())

    def __len__(self) -> int:
        return self.length

    def _image(self, color) -> torch.Tensor:
        img = torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
        img = img.expand(3, self.image_size, self.image_size).clone()
        return img * 2 - 1            # to [-1, 1]

    def __getitem__(self, i: int):
        name = self.names[i % len(self.names)]
        tok = self.tok
        # prompt: <image>... what color is this? -> assistant: a {color} image
        ids: List[int] = [tok.bos_id, tok.token_to_id("<|im_start|>")]
        ids += tok.encode("user\n")
        ids += [self.image_token_id] * self.num_image_tokens
        ids += tok.encode("what color is this?")
        ids += [tok.token_to_id("<|im_end|>"), tok.token_to_id("<|im_start|>")]
        ids += tok.encode("assistant\n")
        prefix_len = len(ids)
        answer = tok.encode(f"a {name} image")
        ids += answer + [tok.token_to_id("<|im_end|>"), tok.eos_id]

        labels = [IGNORE] * prefix_len + answer + [tok.token_to_id("<|im_end|>"), tok.eos_id]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y, self._image(COLORS[name])


def make_vlm_collate(pad_id: int):
    """Pad ids/labels to batch max; stack images. Image tokens sit at the front so
    padding (at the end) never changes their count."""
    def collate(batch):
        maxlen = max(x.size(0) for x, _, _ in batch)
        xs, ys, imgs = [], [], []
        for x, y, img in batch:
            pad = maxlen - x.size(0)
            xs.append(torch.cat([x, torch.full((pad,), pad_id, dtype=torch.long)]))
            ys.append(torch.cat([y, torch.full((pad,), IGNORE, dtype=torch.long)]))
            imgs.append(img)
        return torch.stack(xs), torch.stack(ys), torch.stack(imgs)
    return collate
