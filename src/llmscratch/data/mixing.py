"""Weighted interleaving of multiple document streams (e.g. web text + code).

Pretraining a small model benefits from a *mix*: mostly high-quality web/edu text
plus a little code so the model can read/write code. This yields documents from
several iterables according to weights, deterministically (seeded), dropping a
source when it's exhausted. Used by both prepare_data (.bin) and the streaming loader.
"""
from __future__ import annotations

from typing import Iterable, Iterator, List

import numpy as np


def weighted_interleave(iterables: List[Iterable], weights: List[float],
                        seed: int = 0) -> Iterator:
    """Yield items from `iterables`, picking a source ~ proportional to `weights`.

    Deterministic given seed. When a source is exhausted it's dropped and the
    remaining weights are renormalized.
    """
    if len(iterables) != len(weights):
        raise ValueError("iterables and weights must have equal length")
    rng = np.random.default_rng(seed)
    iters = [iter(it) for it in iterables]
    w = np.asarray(weights, dtype=np.float64)
    if (w <= 0).any():
        raise ValueError("weights must be positive")
    active = list(range(len(iters)))
    while active:
        aw = w[active]
        aw = aw / aw.sum()
        choice = active[int(rng.choice(len(active), p=aw))]
        try:
            yield next(iters[choice])
        except StopIteration:
            active.remove(choice)
