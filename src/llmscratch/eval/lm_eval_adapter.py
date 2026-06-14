"""Adapter so our custom Decoder runs inside EleutherAI lm-evaluation-harness.

Why bother when we already have eval/tasks/? Those give us fast, transparent numbers
for tracking progress. lm-eval-harness gives **official, comparable** numbers (the
exact prompts/normalization used on public leaderboards). Because our model already
exposes the loglikelihood primitive, the adapter is thin: map the harness's three
request types to our scoring/generation.

    pip install lm-eval
    python scripts/lm_eval_run.py --ckpt <ckpt> --tasks hellaswag,arc_easy,gsm8k

`score_pair` below is plain (no lm_eval import) so its logic is unit-testable without
installing the harness; the LM subclass is only defined when lm_eval is present.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def score_pair(model, tokenizer, context: str, continuation: str,
               device: str = "cpu") -> Tuple[float, bool]:
    """Return (sum logprob of continuation | context, is_greedy).

    Uses the harness's pair-encoding convention: tokenize context+continuation
    together and split at len(tokenize(context)) so the boundary is handled
    correctly. is_greedy = every continuation token is the argmax (used for `acc`).
    """
    ctx_enc = tokenizer.encode(context)
    whole_enc = tokenizer.encode(context + continuation)
    cont_enc = whole_enc[len(ctx_enc):] or tokenizer.encode(continuation)

    ids = [tokenizer.bos_id] + ctx_enc + cont_enc
    max_len = model.cfg.max_seq_len
    if len(ids) > max_len:                       # left-truncate, keep continuation
        ids = ids[-max_len:]
    n_cont = min(len(cont_enc), len(ids) - 1)

    x = torch.tensor([ids], device=device)
    logits, _ = model(x)
    logprobs = F.log_softmax(logits[0].float(), dim=-1)

    total, greedy = 0.0, True
    start = len(ids) - n_cont
    for i in range(start, len(ids)):
        total += logprobs[i - 1, ids[i]].item()
        if int(logprobs[i - 1].argmax()) != ids[i]:
            greedy = False
    return total, greedy


def build_lm(ckpt_path: str, tokenizer_path: str, device: str = "cpu",
             max_gen_toks: int = 256):
    """Construct the lm-eval LM object from a checkpoint (requires lm_eval)."""
    from ..model import Decoder, ModelConfig
    from ..tokenizer import build_tokenizer
    from ..utils.checkpoint import load_checkpoint

    tok = build_tokenizer({"mode": "bpe", "path": tokenizer_path})
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Decoder(ModelConfig(**payload["model_config"]))
    load_checkpoint(ckpt_path, model, map_location=device)
    model.to(device).eval()
    return LMScratchLM(model, tok, device=device, max_gen_toks=max_gen_toks)


# ---- the harness LM subclass (only defined if lm_eval is installed) --------
try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model

    @register_model("llmscratch")
    class LMScratchLM(LM):
        def __init__(self, model, tokenizer, device: str = "cpu", max_gen_toks: int = 256):
            super().__init__()
            self.model = model
            self.tok = tokenizer
            self.device = device
            self.max_gen_toks = max_gen_toks

        def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
            return [score_pair(self.model, self.tok, ctx, cont, self.device)
                    for ctx, cont in (r.args for r in requests)]

        def loglikelihood_rolling(self, requests) -> List[float]:
            return [score_pair(self.model, self.tok, "", r.args[0], self.device)[0]
                    for r in requests]

        def generate_until(self, requests) -> List[str]:
            from ..serve.generate import generate
            out = []
            for r in requests:
                context, gen_kwargs = r.args
                until = (gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []) or []
                max_toks = (gen_kwargs.get("max_gen_toks", self.max_gen_toks)
                            if isinstance(gen_kwargs, dict) else self.max_gen_toks)
                ids = [self.tok.bos_id] + self.tok.encode(context)
                gen_ids = generate(self.model, ids, max_new_tokens=max_toks,
                                   temperature=0.0, top_k=None, top_p=None,
                                   stop_ids=[self.tok.eos_id], device=self.device)
                text = self.tok.decode(gen_ids)
                for u in until:
                    if u and u in text:
                        text = text.split(u)[0]
                out.append(text)
            return out

except ImportError:
    class LMScratchLM:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("lm-eval not installed. Run: pip install lm-eval")
