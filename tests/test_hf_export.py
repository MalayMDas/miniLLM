"""The HF Llama export must be numerically identical to our Decoder (validates the
RoPE convention + weight remap). Skipped if transformers isn't installed."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.model import Decoder, ModelConfig

pytest.importorskip("transformers")
from llmscratch.model.hf_export import to_llama_hf, to_llama_config


def test_exported_llama_matches_decoder_logits():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=512, dim=128, n_layers=3, n_heads=8, n_kv_heads=4,
                      max_seq_len=64)
    dec = Decoder(cfg).eval()
    hf = to_llama_hf(dec)
    ids = torch.randint(0, cfg.vocab_size, (2, 20))
    with torch.no_grad():
        ours, _ = dec(ids)
        theirs = hf(ids).logits
    assert (ours - theirs).abs().max().item() < 1e-3


def test_llama_config_dims_match():
    cfg = ModelConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2)
    lc = to_llama_config(cfg)
    assert lc.hidden_size == 64 and lc.num_attention_heads == 4
    assert lc.num_key_value_heads == 2 and lc.vocab_size == 256
    assert lc.tie_word_embeddings is True
