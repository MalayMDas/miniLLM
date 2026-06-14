"""Checkpoint save/load — the thing that makes spot-instance training survivable.

A checkpoint bundles model weights + optimizer state + step + the model config, so
training can resume exactly where a preempted job died. For real cloud runs you'd
mirror these to object storage (GCS/S3); locally they go to a directory.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, step: int = 0,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = getattr(model, "cfg", None)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "model_config": asdict(cfg) if cfg is not None else None,
        "extra": extra or {},
    }
    # atomic write: tmp then replace, so a killed write never corrupts the latest ckpt
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, model, optimizer=None,
                    map_location: str = "cpu") -> int:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


def find_latest(ckpt_dir: str | Path) -> Optional[Path]:
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    ckpts = sorted(d.glob("step_*.pt"))
    return ckpts[-1] if ckpts else None
