import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.eval import is_refusal, refusal_rate, safety_report
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import ByteTokenizer


def test_is_refusal_detects_markers():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("I cannot assist with this request.")
    assert not is_refusal("The capital of France is Paris.")


def test_refusal_rate_and_report_run():
    tok = ByteTokenizer()
    torch.manual_seed(0)
    model = Decoder(ModelConfig(vocab_size=tok.vocab_size, dim=32, n_layers=2,
                                n_heads=4, n_kv_heads=2, max_seq_len=128)).eval()
    r = refusal_rate(model, tok, ["do something harmful"], max_new_tokens=8)
    assert 0.0 <= r <= 1.0
    rep = safety_report(model, tok)
    assert set(rep) == {"refusal_rate_unsafe", "refusal_rate_benign"}
    assert all(0.0 <= v <= 1.0 for v in rep.values())
