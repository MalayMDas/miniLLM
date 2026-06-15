import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.utils import build_logger


def test_wandb_falls_back_to_tensorboard_when_missing(tmp_path):
    # wandb is an optional dep; a wandb-configured run must not crash without it.
    lg = build_logger({"backend": "wandb", "logdir": str(tmp_path)}, "run", {})
    assert type(lg).__name__ in ("TensorBoardLogger", "WandbLogger")  # fallback or real
    lg.log_scalars({"train/loss": 1.0}, 0)
    lg.close()


def test_none_backend_is_noop():
    lg = build_logger({"backend": "none"}, "run", {})
    assert type(lg).__name__ == "NoopLogger"
    lg.log_scalar("x", 1.0, 0)
    lg.close()
