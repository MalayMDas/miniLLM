import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.eval import perplexity, multiple_choice_accuracy, continuation_logprob
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import ByteTokenizer


def _tiny():
    tok = ByteTokenizer()
    torch.manual_seed(0)
    model = Decoder(ModelConfig(vocab_size=tok.vocab_size, dim=32, n_layers=2,
                                n_heads=4, n_kv_heads=2, max_seq_len=128)).eval()
    return tok, model


def test_perplexity_finite_positive():
    tok, model = _tiny()
    ppl = perplexity(model, tok, ["hello world", "the cat sat"])
    assert math.isfinite(ppl) and ppl > 0


def test_mcq_runs_and_in_range():
    tok, model = _tiny()
    examples = [{"question": "2+2=", "choices": ["4", "5"], "answer": 0}]
    acc = multiple_choice_accuracy(model, tok, examples)
    assert 0.0 <= acc <= 1.0


def test_logprob_is_negative():
    tok, model = _tiny()
    lp = continuation_logprob(model, tok.encode("hi "), tok.encode("there"))
    assert lp < 0   # log of a probability
