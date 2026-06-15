"""Export our Decoder to a genuine HuggingFace `LlamaForCausalLM`.

Why Llama (not a custom HF class): our architecture *is* Llama-style (RoPE,
RMSNorm, SwiGLU, GQA, weight tying) and crucially uses HF's `rotate_half` RoPE
convention — so the weights map directly into `transformers.LlamaForCausalLM` with
NO permutation. Exporting as a real Llama means TRL, vLLM, and lm-eval-harness work
natively, and llama.cpp's GGUF converter recognizes the architecture.

    from llmscratch.model.hf_export import export_hf
    export_hf("artifacts/ckpt_sft/step_X.pt", "artifacts/hf_model", tokenizer="artifacts/tok.json")

Requires `transformers`.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import torch

from .decoder import Decoder, ModelConfig


def _ffn_hidden(cfg: ModelConfig) -> int:
    h = int(cfg.ffn_mult * cfg.dim)
    return cfg.multiple_of * ((h + cfg.multiple_of - 1) // cfg.multiple_of)


def to_llama_config(cfg: ModelConfig):
    from transformers import LlamaConfig
    return LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.dim,
        intermediate_size=_ffn_hidden(cfg),
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=1e-5,
        rope_theta=cfg.rope_theta,
        hidden_act="silu",
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
    )


def _remap_state_dict(decoder: Decoder) -> dict:
    """Map our Decoder weights onto HF Llama parameter names (no permutation needed
    because both use the rotate_half RoPE convention)."""
    sd = decoder.state_dict()
    out = {}
    out["model.embed_tokens.weight"] = sd["tok_emb.weight"]
    for i in range(decoder.cfg.n_layers):
        p = f"blocks.{i}."
        q = f"model.layers.{i}."
        out[q + "input_layernorm.weight"] = sd[p + "attn_norm.weight"]
        out[q + "self_attn.q_proj.weight"] = sd[p + "attn.wq.weight"]
        out[q + "self_attn.k_proj.weight"] = sd[p + "attn.wk.weight"]
        out[q + "self_attn.v_proj.weight"] = sd[p + "attn.wv.weight"]
        out[q + "self_attn.o_proj.weight"] = sd[p + "attn.wo.weight"]
        out[q + "post_attention_layernorm.weight"] = sd[p + "ffn_norm.weight"]
        out[q + "mlp.gate_proj.weight"] = sd[p + "ffn.w1.weight"]
        out[q + "mlp.up_proj.weight"] = sd[p + "ffn.w3.weight"]
        out[q + "mlp.down_proj.weight"] = sd[p + "ffn.w2.weight"]
    out["model.norm.weight"] = sd["norm.weight"]
    out["lm_head.weight"] = sd["lm_head.weight"]      # tied to embeddings
    return out


def to_llama_hf(decoder: Decoder):
    """Build a transformers LlamaForCausalLM carrying our weights."""
    from transformers import LlamaForCausalLM
    hf = LlamaForCausalLM(to_llama_config(decoder.cfg))
    missing, unexpected = hf.load_state_dict(_remap_state_dict(decoder), strict=False)
    # only the (tied) lm_head may be reported; nothing structural should be missing
    bad = [m for m in missing if not m.endswith("lm_head.weight")]
    if bad or unexpected:
        raise RuntimeError(f"weight remap mismatch: missing={bad} unexpected={unexpected}")
    hf.tie_weights()
    return hf.eval()


def export_hf(ckpt_path: str, out_dir: str, tokenizer: Optional[str] = None) -> None:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    decoder = Decoder(ModelConfig(**payload["model_config"]))
    decoder.load_state_dict(_strip(payload["model"]))
    hf = to_llama_hf(decoder)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(out)
    if tokenizer:
        _export_tokenizer(tokenizer, out)
    print(f"exported HF Llama -> {out} ({hf.num_parameters()/1e6:.1f}M params)")


def _strip(state: dict) -> dict:
    """Drop any DDP/compile prefixes from a saved state_dict."""
    out = {}
    for k, v in state.items():
        k = k.replace("module.", "").replace("_orig_mod.", "")
        out[k] = v
    return out


def _export_tokenizer(tok_path: str, out_dir: Path) -> None:
    """Copy our byte-level BPE tokenizer.json + minimal config so HF/llama.cpp can read it."""
    import json
    shutil.copy(tok_path, out_dir / "tokenizer.json")
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast",
           "bos_token": "<|bos|>", "eos_token": "<|eos|>", "pad_token": "<|pad|>",
           "unk_token": None, "model_max_length": 1000000000}
    (out_dir / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))
