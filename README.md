# LLM from Scratch (learning project)

A modular, framework-first pipeline to build, train, evaluate, deploy, quantize,
and *use* a small multimodal LLM. See [PLAN.md](PLAN.md) for the full roadmap.

## Status
**Stage 0–2 scaffolded (local-runnable):** tokenizer (byte + byte-level BPE),
compact Llama-style decoder, packed text dataset, end-to-end smoke-train, tests.

## Quickstart (CPU or small GPU, zero downloads)

```bash
pip install -r requirements.txt
pip install -e .          # so `python -m llmscratch...` works anywhere

# 1. train the byte-level BPE tokenizer on the sample corpus
python scripts/train_tokenizer.py --config configs/model_tiny.yaml

# 2. smoke-train: overfit the tiny corpus end-to-end (loss should drop fast)
python scripts/smoke_train.py --config configs/model_tiny.yaml

# 3. SEE the tokenizer tradeoff (UTF-8 bytes vs byte-level BPE)
python -m llmscratch.tokenizer.compare --bpe artifacts/tok.json

# tests
pytest -q
```

To use raw UTF-8 bytes instead of BPE, set `tokenizer.mode: byte` in the config
(then skip step 1). You'll see sequences get ~4x longer — that's the lesson.

## Why byte-level BPE by default?
Raw UTF-8 (256-vocab) needs no training but makes sequences ~4x longer for English,
which ~4x's pretraining cost and shrinks effective context. Byte-level BPE keeps
UTF-8's universality (no OOV, any language/emoji/code) while compressing text.
The `byte` mode is kept so you can measure the difference yourself.

## Layout
```
configs/   per-stage YAML (swap model size / datasets here)
src/llmscratch/{tokenizer,model,data,...}
scripts/   one-command entrypoints
tests/     shape / masking / round-trip tests
infra/gcp/ (later) Terraform for cloud training & serving
```

## Compute plan
Local smoke-train proves the loop for free. Real pretraining targets a cheap
multi-GPU provider (Lambda / RunPod / vast.ai, ~$200–350/day for 8×A100) rather
than GCP on-demand. Use Spot + checkpointing to stay under budget.
