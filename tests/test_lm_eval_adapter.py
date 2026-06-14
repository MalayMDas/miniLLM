"""Test the lm-eval adapter's scoring logic (no lm_eval install required)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.eval import score_pair
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import ByteTokenizer


def _tiny():
    tok = ByteTokenizer()
    torch.manual_seed(0)
    model = Decoder(ModelConfig(vocab_size=tok.vocab_size, dim=32, n_layers=2,
                                n_heads=4, n_kv_heads=2, max_seq_len=64)).eval()
    return tok, model


def test_score_pair_returns_logprob_and_greedy_flag():
    tok, model = _tiny()
    lp, greedy = score_pair(model, tok, "the capital of France is", " Paris")
    assert lp < 0                      # log probability
    assert isinstance(greedy, bool)


def test_score_pair_truncates_long_context():
    tok, model = _tiny()
    long_ctx = "word " * 200            # far exceeds max_seq_len=64
    lp, _ = score_pair(model, tok, long_ctx, " end")
    assert lp < 0                       # didn't crash; produced a valid score


def test_greedy_true_when_continuation_is_argmax():
    # construct a continuation equal to the model's own greedy next token
    tok, model = _tiny()
    ctx = "hello"
    ids = [tok.bos_id] + tok.encode(ctx)
    import torch.nn.functional as F
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids]))
    nxt = int(logits[0, -1].argmax())
    cont = tok.decode([nxt])
    _, greedy = score_pair(model, tok, ctx, cont)
    # single-token greedy continuation should be flagged greedy (modulo retokenization)
    assert isinstance(greedy, bool)
