# LLM-from-Scratch — Interview Cheat Sheet (one page)

A modular, framework-first pipeline to **build → train → align → evaluate → deploy →
quantize → use** a small multimodal LLM. ~300M target, < $500, runs tiny locally.
Full detail: `ARCHITECTURE.md` · run: `RUNBOOK.md`.

## Architecture (one glance)
```
configs/*.yaml ─ drive everything (size, data, logging — no code edits)
   │
DATA ─► TOKENIZER ─► [DECODER]  ◄─ LOGGING (TB/W&B)
(FineWeb   (byte-BPE   RoPE·RMSNorm·SwiGLU·GQA·SDPA, weight-tied
 stream)    32k)        │
                        ▼
                   TRAIN LOOP (grad-accum, cosine LR, bf16, ckpt/resume; DDP via torchrun)
                        │
   ┌──── pretrain ──► SFT(instruct/tools) ──► reasoning(CoT+GRPO) ──┐
   │                                                                ▼
   └──────────────── EVAL (lm-eval-harness + our benchmarks) ── QUANTIZE(int8/GGUF) ── SERVE(vLLM/FastAPI)
                              VISION toggle (from-scratch ViT ↔ SigLIP) · APPS (RAG, ReAct agent)
```

## Pipeline stages
data(HF stream) → tokenizer(32k byte-BPE) → **base pretrain**(DDP) → instruct SFT →
reasoning(CoT distill + GRPO) → tool SFT → eval(every stage) → quantize → serve →
multimodal(LLaVA toggle) → apps(RAG, agent).

## Design decisions — decision → why (one line each)
- **byte-level BPE default** — UTF-8 bytes + learned merges → universal *and* ~4× shorter seqs (4× cheaper). `byte` mode kept to *measure* it.
- **RMSNorm** — cheaper/stabler than LayerNorm. **RoPE** — relative position, extrapolates, no params.
- **SwiGLU** — better quality/param than GELU MLP. **GQA** — fewer KV heads → smaller KV cache → faster inference.
- **Weight tying** (emb=head), **SDPA** (FlashAttention when available, causal mask free).
- **Framework-first + config-driven** — hand-write the model to learn it; lean on libs for plumbing; scale by editing YAML.
- **Vision = config toggle** — from-scratch ViT (learning) ↔ frozen SigLIP (capability).
- **Pluggable logger / toggleable tokenizer & vision** — decoupled from vendors.
- **DDP** for data-parallel (FSDP/DeepSpeed when model doesn't fit one GPU).
- **Reproducibility = lockfile + CUDA-pinned Docker + run stamp** (gitSHA+cfgHash); `-e .` is just dev ergonomics.
- **SkyPilot over Kubeflow/Airflow** — spot + auto-resume, no cluster to operate.
- **Spot + checkpoint-to-bucket + auto-resume** — cheap and preemption-safe.

## Rapid-fire Q&A
- *Why byte-BPE not raw UTF-8?* False choice — BPE is built on UTF-8 bytes; ~4× compression → 4× cheaper + more context.
- *Why GQA / RoPE?* Smaller KV cache, faster inference / relative & extrapolating positions, parameter-free.
- *How reproducible?* Lockfile + CUDA Docker + stamp each run; not the editable install.
- *Need Kubernetes?* No — SkyPilot covers spot + auto-resume; K8s/Kubeflow only at multi-team scale.
- *Scale 300M→7B?* FSDP/DeepSpeed ZeRO-3 sharding + grad checkpointing + more nodes; config + launcher already abstract it.
- *Survive spot preemption?* Checkpoint every N steps to object storage; relaunch → `find_latest` auto-resumes.
- *Vision fully from scratch?* Too compute-heavy for good quality at budget → toggle to SigLIP, train only the projector.
- *Why did LLaVA phase-1 stay flat?* Projector alignment is useless on a *random* LLM — proves you need a pretrained base.
- *Official benchmark numbers?* lm-eval-harness (our `continuation_logprob` == its loglikelihood request); our `benchmark.py` for fast local checks.
- *300M scores?* HellaSwag/ARC-easy/PIQA above chance; MMLU/GSM8K ≈ chance until scale + reasoning.

## Stack (what / alternative)
PyTorch /JAX · HF tokenizers /SentencePiece · HF datasets(streaming) · SDPA /flash-attn ·
DDP→FSDP/DeepSpeed /Megatron · TensorBoard /W&B,MLflow · lm-eval-harness /HELM ·
vLLM /TGI,llama.cpp · GGUF/AWQ/bitsandbytes · FAISS /Chroma · SkyPilot /Kubeflow · Docker.

## Benchmarks (all local, no API; one-time HF download)
HellaSwag·OpenBookQA (MCQ loglik `acc_norm`) · GSM8K (gen exact-match) · BFCL (tool-call AST match) · VQAv2 (soft-acc).
`python scripts/benchmark.py --ckpt … --tasks hellaswag,openbookqa,gsm8k,bfcl` ·
`python scripts/lm_eval_run.py --ckpt … --tasks hellaswag,arc_easy,piqa,…` (official).

## Cloud run (RUNBOOK.md), spot 8×A100 ≈ $220–380
```
bash infra/cloud_setup.sh
python scripts/train_tokenizer.py --config configs/tokenizer_32k.yaml
torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain_300m.yaml   # auto-resume on preemption
python scripts/lm_eval_run.py --ckpt /artifacts/.../step_X.pt --tokenizer artifacts/tok_32k.json
python scripts/sft.py --config configs/sft_300m.yaml   # → reasoning → tools → quantize → serve
sky down llm-train
```

## Bugs found (memorable)
- `itertools.cycle` on a DataLoader **caches epoch 1** (kills reshuffle / OOMs streaming) → re-iterating loader.
- `.gitignore` **ignores inline comments** → `artifacts/ # x` never matched; checkpoints leaked.
- **torch-before-pyarrow segfaults** on Windows → import `datasets` first.
- CRLF in `.sh` breaks bash on Linux → `.gitattributes` forces LF (Windows-dev→Linux-cloud).
- `torchrun` needs libuv on Windows → verify DDP via `mp.spawn`+FileStore (`check_ddp.py`).

## Numbers to remember
~300M params · 24L×1024d, 16 q-heads / 4 kv-heads · ctx 2048 · 32k vocab · ~10–20B tokens (Chinchilla-ish) ·
tiny local = ~1M params on CPU · **35 tests** · int8 ≈ 2.9× smaller, ~0% ppl change · byte-BPE ≈ 4× vs UTF-8.
