# LLM from Scratch (learning project)

A modular, framework-first pipeline to build, train, evaluate, deploy, quantize,
and *use* a small multimodal LLM. See [PLAN.md](PLAN.md) for the full roadmap,
[ARCHITECTURE.md](ARCHITECTURE.md) for design decisions + the file-by-file map, and
[RUNBOOK.md](RUNBOOK.md) for the step-by-step **cloud training run** (provision →
tokenizer → pretrain → SFT → reasoning → eval → quantize/serve, with cost controls).

## Status
**All stages built and locally runnable** (tiny configs; scale via YAML):
tokenizer (byte + byte-level BPE) · Llama-style decoder · base pretraining
(local or HF streaming, spot-safe resume) · instruct SFT · reasoning (CoT + GRPO)
· tool use · evaluation (perplexity + multiple-choice) · OpenAI-compatible serving
· int8 quantization · multimodal vision toggle (from-scratch ViT ↔ SigLIP) · apps
(RAG, ReAct agent). **27 tests passing.** Cloud scaffolding: Dockerfile, lockfile,
SkyPilot, setup script. Remaining work is *scaling up* (real data + multi-GPU), not
new components.

### Stage entrypoints
```bash
python scripts/pretrain.py --config configs/pretrain_tiny.yaml   # base model
python scripts/sft.py      --config configs/sft_tiny.yaml        # instruct
python scripts/quantize.py --ckpt artifacts/ckpt_pretrain/step_0000200.pt
```

## Quickstart (CPU or small GPU, zero downloads)

```bash
pip install -r requirements.txt
pip install -e .          # so `python -m llmscratch...` works anywhere

# 1. train the byte-level BPE tokenizer on the sample corpus
python scripts/train_tokenizer.py --config configs/model_tiny.yaml

# 2. smoke-train: overfit the tiny corpus end-to-end (loss should drop fast)
python scripts/smoke_train.py --config configs/model_tiny.yaml

# 3. watch progress in TensorBoard (loss, perplexity, lr, grad-norm, tok/s, samples)
tensorboard --logdir runs
# open http://localhost:6006   (Scalars + Text tabs)

# 4. SEE the tokenizer tradeoff (UTF-8 bytes vs byte-level BPE)
python -m llmscratch.tokenizer.compare --bpe artifacts/tok.json

# tests
pytest -q
```

To use raw UTF-8 bytes instead of BPE, set `tokenizer.mode: byte` in the config
(then skip step 1). You'll see sequences get ~4x longer — that's the lesson.

## Verify it works locally (no GPU, no downloads)

Two complementary checks — this is the fastest way to confirm a fresh clone is healthy:

```bash
# A. SEE every stage run end-to-end in a few seconds (prints what each did)
python scripts/demo.py
#    tokenizer -> eval -> tool use -> RAG -> agent -> vision -> quantization

# B. RUN the test suite (checks invariants: causality, loss-masking,
#    tool round-trips, multimodal fusion, scoring, RAG relevance)
pytest                       # 27 tests; pass after `pip install -e .`
```

Then exercise the **training** stages (still CPU-friendly at tiny scale):

```bash
python scripts/pretrain.py --config configs/pretrain_tiny.yaml          # base model + checkpoints
python scripts/sft.py      --config configs/sft_tiny.yaml               # instruct (loads the base ckpt)
python scripts/sft.py      --config configs/sft_tools_tiny.yaml         # tool-use SFT
python scripts/train_vision.py --phase 2                               # multimodal (synthetic data)
python scripts/evaluate.py --ckpt artifacts/ckpt_pretrain/step_0000200.pt  # quick local smoke eval
python scripts/benchmark.py --ckpt artifacts/ckpt_pretrain/step_0000200.pt \
    --tasks hellaswag,openbookqa,gsm8k,bfcl --limit 100   # real benchmarks, local, no API
python scripts/quantize.py --ckpt artifacts/ckpt_pretrain/step_0000200.pt  # size/quality report
tensorboard --logdir runs                                              # watch training curves
```

**Multi-GPU** (Linux + NVIDIA) — same script, scaled by `torchrun`:
```bash
torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain_300m.yaml
python scripts/check_ddp.py     # verify the distributed path locally (2 CPU ranks)
```

> The toy model is tiny and (in the demo) untrained, so generated *text* is
> gibberish — by design. What's being verified is that every **mechanism** runs
> correctly; scaling up the config + data is what produces quality.

## Why byte-level BPE by default?
Raw UTF-8 (256-vocab) needs no training but makes sequences ~4x longer for English,
which ~4x's pretraining cost and shrinks effective context. Byte-level BPE keeps
UTF-8's universality (no OOV, any language/emoji/code) while compressing text.
The `byte` mode is kept so you can measure the difference yourself.

## Layout
```
configs/   per-stage YAML (swap model size / datasets here)
src/llmscratch/{tokenizer,model,data,train,align,tools,eval,serve,quantize,vision,apps,utils}
scripts/   one-command entrypoints (demo, train_tokenizer, smoke_train, pretrain, sft, quantize)
tests/     invariant tests (shapes, masking, causality, tools, eval, RAG, vision)
infra/     Dockerfile + cloud_setup.sh + SkyPilot (see infra/README.md)
```
Full file-by-file map and design rationale: [ARCHITECTURE.md](ARCHITECTURE.md).

## Compute plan
Local smoke-train proves the loop for free. Real pretraining targets a cheap
multi-GPU provider (Lambda / RunPod / vast.ai, ~$200–350/day for 8×A100) rather
than GCP on-demand. Use Spot + checkpointing to stay under budget.
