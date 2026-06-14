import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.align import (format_cot_turn, extract_answer, cot_conversation,
                              grpo_step, GRPOConfig)
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import ByteTokenizer


def test_cot_format_and_extract():
    turn = format_cot_turn("2+2 is 4", "The answer is 4.")
    assert turn.startswith("<think>") and "</think>" in turn
    assert extract_answer(turn) == "The answer is 4."


def test_cot_conversation_roles():
    msgs = cot_conversation("q", "r", "a", system="sys")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_grpo_step_runs():
    tok = ByteTokenizer()
    torch.manual_seed(0)
    model = Decoder(ModelConfig(vocab_size=tok.vocab_size, dim=32, n_layers=2,
                                n_heads=4, n_kv_heads=2, max_seq_len=64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # verifiable reward: prefer completions containing the letter 'a'
    reward_fn = lambda prompt, comp: float(comp.count("a"))
    cfg = GRPOConfig(group_size=4, max_new_tokens=8, device="cpu")
    out = grpo_step(model, tok, ["say something:"], reward_fn, opt, cfg)
    assert math.isfinite(out["loss"]) and out["mean_reward"] >= 0
