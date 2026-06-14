"""GRPO — Group Relative Policy Optimization (compact, teaching implementation).

The idea that made RL for reasoning cheap: instead of a learned value/critic network
(as in PPO), sample a GROUP of G completions per prompt and use the group's mean
reward as the baseline. The advantage is the reward standardized within its group:

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

then do a policy-gradient update pushing up the log-prob of above-average completions
and down the below-average ones. Rewards are typically *verifiable* (e.g. "is the
final answer correct?"), so no reward model is needed.

This minimal version implements group-normalized REINFORCE (the core of GRPO). Full
GRPO additionally uses a PPO-style clipped ratio against the pre-update policy and a
KL penalty to a frozen reference model — noted inline where they'd go. Honest scale
note: at ~1M params RL gains are marginal; this is here to make the mechanism
concrete and runnable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

RewardFn = Callable[[str, str], float]      # (prompt, completion_text) -> reward


@dataclass
class GRPOConfig:
    group_size: int = 8
    max_new_tokens: int = 24
    temperature: float = 1.0
    lr: float = 1e-4
    kl_coef: float = 0.0        # set >0 to penalize divergence from the ref policy
    device: str = "cpu"


@torch.no_grad()
def _sample(model, prompt_ids: List[int], cfg: GRPOConfig) -> List[int]:
    idx = torch.tensor([prompt_ids], device=cfg.device)
    new = []
    for _ in range(cfg.max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.max_seq_len:])
        probs = F.softmax(logits[:, -1, :].float() / cfg.temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)
        tok = int(nxt.item())
        new.append(tok)
        idx = torch.cat([idx, nxt], dim=1)
    return new


def _logprob_sum(model, prompt_ids: List[int], comp_ids: List[int],
                 cfg: GRPOConfig) -> torch.Tensor:
    """Differentiable sum of log p(completion | prompt) under the current policy."""
    ids = torch.tensor([prompt_ids + comp_ids], device=cfg.device)
    logits, _ = model(ids)
    logprobs = F.log_softmax(logits[0].float(), dim=-1)
    start = len(prompt_ids)
    idxs = torch.tensor(comp_ids, device=cfg.device)
    positions = torch.arange(start - 1, start - 1 + len(comp_ids), device=cfg.device)
    return logprobs[positions, idxs].sum()


def grpo_step(model, tokenizer, prompts: List[str], reward_fn: RewardFn,
              optimizer, cfg: GRPOConfig) -> dict:
    model.train()
    losses, all_rewards = [], []
    optimizer.zero_grad(set_to_none=True)

    for prompt in prompts:
        prompt_ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
        comps = [_sample(model, prompt_ids, cfg) for _ in range(cfg.group_size)]
        rewards = torch.tensor(
            [reward_fn(prompt, tokenizer.decode(c)) for c in comps], dtype=torch.float32)
        all_rewards.append(rewards.mean().item())

        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)   # group-relative
        for comp, a in zip(comps, adv):
            if len(comp) == 0:
                continue
            lp = _logprob_sum(model, prompt_ids, comp, cfg)
            # maximize advantage-weighted log-prob => minimize its negative.
            # (full GRPO: clip ratio vs old policy + add cfg.kl_coef * KL(policy||ref))
            losses.append(-(a.to(cfg.device) * lp / len(comp)))

    loss = torch.stack(losses).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": loss.item(), "mean_reward": sum(all_rewards) / len(all_rewards)}
