"""Config loading + run provenance helpers.

Keeps the "stamp every run" best practice in one place: a run id derived from the
git SHA + a hash of the resolved config, so any checkpoint can be traced back to the
exact code and settings that produced it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_sha(short: bool = True) -> str:
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if short:
            args = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def config_hash(cfg: Dict[str, Any], n: int = 8) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:n]


def run_id(cfg: Dict[str, Any], prefix: str = "run") -> str:
    """Stable, traceable id: <prefix>-<gitSHA>-<cfgHash>."""
    return f"{prefix}-{git_sha()}-{config_hash(cfg)}"


def pick_device(name: str = "auto") -> str:
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name
