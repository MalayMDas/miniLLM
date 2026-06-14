"""Verify the distributed Trainer path on CPU without torchrun.

    python scripts/check_ddp.py

Spawns 2 gloo ranks via a FileStore (avoids the Windows libuv/TCPStore issue), runs
a few DDP steps over a DistributedSampler, and confirms only rank 0 writes the
checkpoint. On Linux GPU you'd instead launch with torchrun + NCCL (see
utils/distributed.py) — this just proves the Trainer's DDP branches are correct.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.data import iter_local_lines, encode_corpus, PackedDataset
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import ByteTokenizer
from llmscratch.train import Trainer, TrainArgs


def worker(rank: int, world_size: int, init_file: str, ckpt_dir: str):
    dist.init_process_group(backend="gloo", init_method=f"file:///{init_file}",
                            rank=rank, world_size=world_size)
    tok = ByteTokenizer()
    torch.manual_seed(0)
    model = Decoder(ModelConfig(vocab_size=tok.vocab_size, dim=32, n_layers=2,
                                n_heads=4, n_kv_heads=2, max_seq_len=64))
    model = DDP(model)

    ids = encode_corpus(tok, iter_local_lines("data/sample.txt"))
    ds = PackedDataset(ids, 32)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, drop_last=True)
    loader = DataLoader(ds, batch_size=2, sampler=sampler, drop_last=True)

    targs = TrainArgs(steps=10, warmup_steps=2, lr=6e-4, grad_accum=2, device="cpu",
                      amp=False, log_every=5, ckpt_every=0, ckpt_dir=ckpt_dir,
                      is_main=(rank == 0))
    Trainer(model, loader, targs).train()
    dist.destroy_process_group()


def main():
    init_file = Path(tempfile.gettempdir()) / "llmscratch_ddp_store"
    init_file.unlink(missing_ok=True)
    ckpt_dir = Path(tempfile.gettempdir()) / "llmscratch_ddp_ckpt"
    mp.spawn(worker, args=(2, init_file.as_posix(), str(ckpt_dir)), nprocs=2, join=True)

    ckpts = list(ckpt_dir.glob("step_*.pt"))
    print(f"\nrank-0 checkpoints written: {[c.name for c in ckpts]}")
    assert len(ckpts) == 1, "expected exactly one checkpoint (only rank 0 should write)"
    print("DDP path OK: 2 ranks trained, single checkpoint from rank 0.")


if __name__ == "__main__":
    main()
