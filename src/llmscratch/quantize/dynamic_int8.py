"""Post-training dynamic int8 quantization (the simplest, dependency-free option).

Quantizes Linear weights to int8 and dequantizes on the fly. Runs on CPU, needs no
calibration data, and shrinks the model ~4x. It's the easy on-ramp; the trade-off
ladder for real local deployment:

  - dynamic int8 (here)      : trivial, CPU, modest speedup, small quality loss.
  - bitsandbytes 4/8-bit     : quick GPU load in low VRAM, good for experiments.
  - GPTQ / AWQ (4-bit)       : calibrated, best quality-per-bit for GPU serving (vLLM).
  - GGUF + llama.cpp         : the "runs on any laptop / CPU / Apple Silicon" path.

Best practice: always report quality (perplexity) alongside the size/speed win —
quantization is a trade, not a free lunch.
"""
from __future__ import annotations

import copy
import io

import torch
import torch.nn as nn


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Return an int8-quantized copy (Linear layers). CPU inference."""
    model = copy.deepcopy(model).cpu().eval()
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def serialized_size_bytes(model: nn.Module) -> int:
    """On-disk size of the model's state_dict — works for fp32 and quantized
    modules alike (quantized Linear weights are packed, not in .parameters())."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes
