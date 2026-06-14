"""A small, transparent training loop reused by pretraining and SFT.

Deliberately readable rather than clever. It covers the things that actually matter
in practice: gradient accumulation, cosine LR with warmup, gradient clipping, mixed
precision, periodic eval, checkpoint/resume, and pluggable logging.

Scaling note: for multi-GPU you wrap the model in DDP/FSDP and launch with
`torchrun` / `accelerate` — the loop body is unchanged. This single-process version
runs anywhere (CPU or one GPU) so every stage is locally testable.
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from ..utils.checkpoint import save_checkpoint
from ..utils.metrics_logger import Logger, NoopLogger


@dataclass
class TrainArgs:
    steps: int = 1000
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_accum: int = 1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    log_every: int = 10
    eval_every: int = 0          # 0 disables eval
    ckpt_every: int = 0          # 0 disables checkpointing
    ckpt_dir: str = "artifacts/ckpt"
    amp: bool = True             # bf16 autocast on cuda
    device: str = "cuda"
    is_main: bool = True         # only the main DDP process logs / checkpoints
    time_budget_min: Optional[float] = None  # wall-clock stop (e.g. for a 2h local run)
    lr_offset: int = 0           # anchor warmup+cosine here (set to resume step when
                                 # extending a finished model, so it gets a fresh schedule)


def _infinite(loader):
    """Re-iterate a loader forever. Unlike itertools.cycle this re-creates the
    iterator each epoch — so map-style datasets reshuffle and IterableDatasets
    (streaming) keep streaming instead of being cached in memory."""
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch
        if empty:
            raise RuntimeError("train loader produced 0 batches — check batch_size "
                               "vs dataset size (drop_last may be dropping everything)")


def _lr_at(step: int, a: TrainArgs) -> float:
    # schedule is measured from lr_offset (0 normally; = resume step when extending)
    s = step - a.lr_offset
    total = a.steps - a.lr_offset
    if s < a.warmup_steps:
        return a.lr * (s + 1) / max(1, a.warmup_steps)
    prog = (s - a.warmup_steps) / max(1, total - a.warmup_steps)
    prog = min(1.0, prog)
    coeff = 0.5 * (1 + math.cos(math.pi * prog))
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * coeff)


class Trainer:
    def __init__(self, model, train_loader: DataLoader, args: TrainArgs,
                 logger: Optional[Logger] = None,
                 eval_fn: Optional[Callable[[], dict]] = None,
                 tokens_per_step: Optional[int] = None):
        self.model = model.to(args.device)
        # unwrap DDP/FSDP for checkpointing + config access
        self.raw_model = model.module if hasattr(model, "module") else model
        self.loader = train_loader
        self.a = args
        self.logger = logger or NoopLogger()
        self.eval_fn = eval_fn
        self.tokens_per_step = tokens_per_step
        # AdamW with no weight decay on 1-D params (norms/biases) — standard recipe.
        decay, no_decay = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": args.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=args.betas)
        self.use_amp = args.amp and args.device.startswith("cuda")

    def train(self, start_step: int = 0) -> None:
        a = self.a
        if start_step >= a.steps:
            if a.is_main:
                print(f"already at step {start_step} >= target {a.steps}; nothing to do. "
                      f"Use --add-steps to train further.")
            return
        self.model.train()
        data = _infinite(self.loader)
        run_start = time.perf_counter()
        last_step = start_step
        for step in range(start_step, a.steps):
            last_step = step
            lr = _lr_at(step, a)
            for g in self.opt.param_groups:
                g["lr"] = lr

            t0 = time.perf_counter()
            self.opt.zero_grad(set_to_none=True)
            loss_val = 0.0
            for micro in range(a.grad_accum):
                x, y = next(data)
                x, y = x.to(a.device), y.to(a.device)
                # under DDP, only sync grads on the last micro-step (avoids
                # redundant all-reduce during accumulation)
                last = micro == a.grad_accum - 1
                sync_ctx = (self.model.no_sync() if (not last and hasattr(self.model, "no_sync"))
                            else _nullcontext())
                with sync_ctx:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                        _, loss = self.model(x, y)
                        loss = loss / a.grad_accum
                    loss.backward()
                loss_val += loss.item()
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), a.grad_clip)
            self.opt.step()
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            if a.is_main and (step % a.log_every == 0 or step == a.steps - 1):
                metrics = {
                    "train/loss": loss_val,
                    "train/perplexity": math.exp(min(loss_val, 20)),
                    "train/lr": lr,
                    "train/grad_norm": float(gnorm),
                    "perf/step_time_ms": dt * 1e3,
                }
                if self.tokens_per_step:
                    metrics["perf/tokens_per_sec"] = self.tokens_per_step * a.grad_accum / max(dt, 1e-6)
                self.logger.log_scalars(metrics, step)
                print(f"step {step:5d} | loss {loss_val:.4f} | lr {lr:.2e}")

            if a.is_main and a.eval_every and self.eval_fn and step > 0 and step % a.eval_every == 0:
                self.model.eval()
                with torch.no_grad():
                    self.logger.log_scalars({f"eval/{k}": v for k, v in self.eval_fn().items()}, step)
                self.model.train()

            if a.is_main and a.ckpt_every and step > 0 and step % a.ckpt_every == 0:
                save_checkpoint(Path(a.ckpt_dir) / f"step_{step:07d}.pt",
                                self.raw_model, self.opt, step)

            if a.time_budget_min is not None and (time.perf_counter() - run_start) / 60 >= a.time_budget_min:
                if a.is_main:
                    print(f"[time budget {a.time_budget_min} min reached at step {step}]")
                break

        final = last_step + 1
        if a.is_main:
            save_checkpoint(Path(a.ckpt_dir) / f"step_{final:07d}.pt",
                            self.raw_model, self.opt, final)
        self.logger.close()
