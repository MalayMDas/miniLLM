"""Supervised fine-tuning (SFT) — turn a base model into an instruction follower.

We render conversations into ChatML, mask everything except assistant turns, and
train next-token prediction on the unmasked part. Production stacks use TRL's
SFTTrainer; this transparent version uses our own Trainer so the mechanics (loss
masking, shifting) are visible and locally runnable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..data.chat import render_chat, IGNORE


def load_chat_jsonl(path: str | Path) -> List[List[Dict[str, str]]]:
    """Each line: {"messages": [{"role": ..., "content": ...}, ...]}."""
    convs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(json.loads(line)["messages"])
    return convs


class SFTDataset(Dataset):
    def __init__(self, tokenizer, conversations: List[List[Dict[str, str]]],
                 max_len: int = 1024):
        self.tok = tokenizer
        self.convs = conversations
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.convs)

    def __getitem__(self, i: int):
        input_ids, labels = render_chat(self.tok, self.convs[i])
        input_ids = input_ids[: self.max_len + 1]
        labels = labels[: self.max_len + 1]
        # shift so the model predicts token t+1 from tokens <= t (matches Decoder loss)
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y


def make_collate(pad_id: int):
    """Right-pad a batch to its longest sequence; pad labels with IGNORE."""
    def collate(batch):
        maxlen = max(x.size(0) for x, _ in batch)
        xs, ys = [], []
        for x, y in batch:
            pad = maxlen - x.size(0)
            xs.append(torch.cat([x, torch.full((pad,), pad_id, dtype=torch.long)]))
            ys.append(torch.cat([y, torch.full((pad,), IGNORE, dtype=torch.long)]))
        return torch.stack(xs), torch.stack(ys)
    return collate
