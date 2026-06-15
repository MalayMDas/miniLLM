"""Pluggable experiment logger.

Training code calls a tiny, backend-agnostic interface (`log_scalar`, `log_text`,
`log_hparams`). Backends are swappable so we are NOT coupled to one tool:
  - 'tensorboard' : local, zero-account, great for a single box (default).
  - 'wandb'       : hosted, team dashboards, sweeps (added in the cloud stage).
  - 'none'        : disable logging (e.g. unit tests).

This mirrors how real stacks keep the trainer independent of the tracking tool.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class Logger:
    def log_scalar(self, tag: str, value: float, step: int) -> None: ...
    def log_scalars(self, mapping: Dict[str, float], step: int) -> None:
        for k, v in mapping.items():
            self.log_scalar(k, v, step)
    def log_text(self, tag: str, text: str, step: int) -> None: ...
    def log_hparams(self, hparams: Dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class NoopLogger(Logger):
    def log_scalar(self, tag, value, step): pass
    def log_text(self, tag, text, step): pass
    def log_hparams(self, hparams): pass
    def close(self): pass


class TensorBoardLogger(Logger):
    def __init__(self, logdir: str | Path):
        import json
        from torch.utils.tensorboard import SummaryWriter
        self._json = json
        self.logdir = str(logdir)
        Path(self.logdir).mkdir(parents=True, exist_ok=True)
        self.w = SummaryWriter(self.logdir)
        # plain-text mirror so you can tail loss live without opening TensorBoard
        self._jsonl = open(Path(self.logdir) / "metrics.jsonl", "a", encoding="utf-8")

    def _write_jsonl(self, mapping: Dict[str, float], step: int) -> None:
        self._jsonl.write(self._json.dumps({"step": step, **mapping}) + "\n")
        self._jsonl.flush()                  # flush so `tail` sees it immediately

    def log_scalar(self, tag, value, step):
        self.w.add_scalar(tag, value, step)
        self._write_jsonl({tag: value}, step)

    def log_scalars(self, mapping, step):
        for k, v in mapping.items():
            self.w.add_scalar(k, v, step)
        self._write_jsonl(dict(mapping), step)

    def log_text(self, tag, text, step):
        # markdown code-fence so newlines render in the TB Text tab
        self.w.add_text(tag, f"```\n{text}\n```", step)

    def log_hparams(self, hparams):
        # flatten nested dicts to scalars/strings TB accepts
        flat = _flatten(hparams)
        self.w.add_text("hparams", "\n".join(f"{k}: {v}" for k, v in flat.items()), 0)

    def close(self):
        self.w.flush()
        self.w.close()
        self._jsonl.close()


class WandbLogger(Logger):
    """Hosted backend. Requires `pip install wandb` and a login/API key."""
    def __init__(self, project: str, run_name: Optional[str], config: Dict[str, Any]):
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name, config=config)

    def log_scalar(self, tag, value, step):
        self.wandb.log({tag: value}, step=step)

    def log_scalars(self, mapping, step):
        self.wandb.log(mapping, step=step)

    def log_text(self, tag, text, step):
        self.wandb.log({tag: self.wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def log_hparams(self, hparams):
        self.run.config.update(_flatten(hparams), allow_val_change=True)

    def close(self):
        self.run.finish()


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def build_logger(cfg: Dict[str, Any], run_name: str, config: Dict[str, Any]) -> Logger:
    """cfg keys: backend ('tensorboard'|'wandb'|'none'), logdir, project."""
    backend = cfg.get("backend", "tensorboard")
    if backend == "none":
        return NoopLogger()
    if backend == "wandb":
        # wandb is optional: fall back to TensorBoard if it isn't installed, so a
        # cloud config doesn't crash a local run that lacks the dependency.
        try:
            return WandbLogger(cfg.get("project", "llm-from-scratch"), run_name, config)
        except ImportError:
            print("[logger] wandb not installed -> using TensorBoard "
                  "(`pip install wandb` for hosted tracking).")
            backend = "tensorboard"
    if backend == "tensorboard":
        logdir = Path(cfg.get("logdir", "runs")) / run_name
        return TensorBoardLogger(logdir)
    raise ValueError(f"unknown logger backend: {backend}")
