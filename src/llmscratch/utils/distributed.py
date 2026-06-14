"""Distributed-training helpers (data parallel via torch.distributed).

The single-GPU loop and the multi-GPU loop share the same Trainer; this module
just handles process-group setup and "am I the main process?" gating so only rank 0
logs and checkpoints.

Launch (Linux cloud):
    torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain_300m.yaml

DDP (data parallel) is the right tool when the model fits on one GPU. For models
that don't fit, switch the backend to **FSDP** / **DeepSpeed ZeRO** (shards params,
grads, optimizer state across GPUs) — easiest via `accelerate config`; the loop body
is unchanged. Kept as DDP here because it's transparent and testable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_main: bool = True
    device: str = "cpu"

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed() -> DistInfo:
    """Initialize the process group if launched under torchrun; else single process."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return DistInfo(device=device)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank,
                    is_main=(rank == 0), device=device)


def cleanup() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
