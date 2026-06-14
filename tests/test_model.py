import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.model import Decoder, ModelConfig


def tiny_cfg(**kw):
    base = dict(vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shapes():
    cfg = tiny_cfg()
    model = Decoder(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss.dim() == 0 and loss.item() > 0


def test_gqa_divisibility():
    import pytest
    with pytest.raises(AssertionError):
        Decoder(tiny_cfg(n_heads=4, n_kv_heads=3))


def test_generate_grows_sequence():
    cfg = tiny_cfg()
    model = Decoder(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(x, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_causality_future_tokens_dont_change_past():
    # Changing a later token must not alter logits at earlier positions.
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = Decoder(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        l1, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size
        l2, _ = model(x2)
    assert torch.allclose(l1[0, :-1], l2[0, :-1], atol=1e-5)
