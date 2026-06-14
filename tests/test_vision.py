import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import ByteTokenizer
from llmscratch.vision import ViT, ViTConfig, MultimodalDecoder, build_vision_encoder


def _build():
    tok = ByteTokenizer()
    torch.manual_seed(0)
    dim = 32
    decoder = Decoder(ModelConfig(vocab_size=tok.vocab_size, dim=dim, n_layers=2,
                                  n_heads=4, n_kv_heads=2, max_seq_len=64))
    vit = ViT(ViTConfig(image_size=16, patch_size=8, dim=dim, depth=2, heads=2))  # 4 patches
    mm = MultimodalDecoder(decoder, vit, image_token_id=tok.token_to_id("<image>"))
    return tok, mm, vit.num_tokens


def _make_batch(tok, n_img_tokens, B=2):
    img = tok.token_to_id("<image>")
    text = tok.encode("describe this")
    ids = [img] * n_img_tokens + text
    input_ids = torch.tensor([ids] * B)
    pixels = torch.randn(B, 3, 16, 16)
    return input_ids, pixels


def test_vit_output_shape():
    vit = ViT(ViTConfig(image_size=16, patch_size=8, dim=32, depth=2, heads=2))
    out = vit(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 4, 32)


def test_multimodal_forward_and_loss():
    tok, mm, n = _build()
    input_ids, pixels = _make_batch(tok, n)
    logits, loss = mm(input_ids, pixels, targets=input_ids)
    assert logits.shape[0] == 2 and logits.shape[-1] == tok.vocab_size
    assert loss.item() > 0


def test_image_changes_output():
    # different pixels must produce different logits (vision is actually wired in)
    tok, mm, n = _build()
    mm.eval()
    input_ids, pixels = _make_batch(tok, n)
    with torch.no_grad():
        l1, _ = mm(input_ids, pixels)
        l2, _ = mm(input_ids, pixels + 1.0)
    assert not torch.allclose(l1, l2)


def test_wrong_image_token_count_raises():
    tok, mm, n = _build()
    import pytest
    input_ids, pixels = _make_batch(tok, n - 1)   # one too few placeholders
    with pytest.raises(ValueError):
        mm(input_ids, pixels)


def test_encoder_toggle_from_scratch():
    enc = build_vision_encoder({"mode": "from_scratch", "image_size": 16,
                                "patch_size": 8, "dim": 32, "depth": 1, "heads": 2})
    assert enc.num_tokens == 4 and enc.out_dim == 32
