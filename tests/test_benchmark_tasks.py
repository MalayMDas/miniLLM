"""Unit-test benchmark SCORING logic on fixtures (no dataset downloads)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.eval.tasks import (extract_pred, gold_answer, call_matches,
                                   vqa_accuracy, normalize_answer, evaluate_bfcl)
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import ByteTokenizer


# ---- GSM8K answer parsing -------------------------------------------------
def test_gsm8k_gold_and_pred_extraction():
    assert gold_answer("blah blah #### 72") == "72"
    assert gold_answer("steps #### 1,200") == "1200"
    assert extract_pred("So the total is 72. #### 72") == "72"
    assert extract_pred("the answer is 1,200 dollars") == "1200"
    assert extract_pred("no number here") is None


# ---- BFCL AST match -------------------------------------------------------
def test_bfcl_call_match():
    gold = {"name": "calculator", "arguments": {"expression": "6*7"}}
    assert call_matches({"name": "calculator", "arguments": {"expression": "6*7"}}, gold)
    assert not call_matches({"name": "search", "arguments": {"expression": "6*7"}}, gold)
    assert not call_matches({"name": "calculator", "arguments": {"expression": "7*6"}}, gold)
    assert not call_matches(None, gold)


# ---- VQA accuracy ---------------------------------------------------------
def test_vqa_normalization_and_accuracy():
    assert normalize_answer("The Two dogs!") == "2 dogs"
    answers = ["cat"] * 3 + ["dog"] * 7
    assert vqa_accuracy("cat", answers) == 1.0          # 3 matches -> capped at 1.0
    assert vqa_accuracy("dog", answers) == 1.0          # 7 matches
    assert vqa_accuracy("fish", answers) == 0.0


def test_bfcl_evaluate_runs_end_to_end():
    # tiny untrained model; just verify the eval pipeline returns a valid score
    tok = ByteTokenizer()
    torch.manual_seed(0)
    model = Decoder(ModelConfig(vocab_size=tok.vocab_size, dim=32, n_layers=2,
                                n_heads=4, n_kv_heads=2, max_seq_len=128)).eval()
    examples = [{"question": "what is 6 times 7?",
                 "tools": [{"name": "calculator", "parameters": {"expression": {"type": "string"}}}],
                 "gold": {"name": "calculator", "arguments": {"expression": "6*7"}}}]
    acc = evaluate_bfcl(model, tok, examples, device="cpu", max_new_tokens=8)
    assert 0.0 <= acc <= 1.0
