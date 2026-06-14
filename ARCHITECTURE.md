# Architecture & Design (interview prep)

A from-scratch, modular pipeline to build / train / evaluate / deploy / use a small
multimodal LLM. This doc explains **what each piece is, why it was chosen over the
alternatives, and the trade-offs** — written so you can defend every decision.

> Philosophy: *framework-first, config-driven, easy to run locally, reproducible in
> the cloud.* Small enough to train on a budget (<$500), structured like a real stack.

---

## 1. Top-level architecture

```
                         configs/*.yaml  (one knob-set per stage; swap model size, data, logging)
                                │
        ┌───────────────────────┼───────────────────────────────────────────┐
        ▼                       ▼                                             ▼
   ┌─────────┐           ┌──────────────┐                              ┌────────────┐
   │  DATA   │  tokens   │  TOKENIZER   │   ids                         │  LOGGING   │
   │ text.py │──────────▶│ byte | BPE   │──────────┐                    │ TB | W&B   │
   └─────────┘           └──────────────┘          │                    └─────▲──────┘
   (local .txt now;                                 ▼                          │ scalars/text
    HF streaming later)                       ┌──────────┐  loss/logits        │
                                              │  MODEL   │─────────────────────┘
                                              │ decoder  │  (RoPE/RMSNorm/
                                              │  .py     │   SwiGLU/GQA)
                                              └────┬─────┘
                                                   │ AdamW + cosine LR
                                        ┌──────────▼───────────┐
                                        │   TRAIN LOOP          │  scripts/smoke_train.py
                                        │  (overfit smoke now;  │  → checkpoints + samples
                                        │   FSDP/DeepSpeed next)│
                                        └──────────┬───────────┘
                                                   │
        ┌──────────────────────────┬──────────────┴───────────────┬─────────────────────┐
        ▼                          ▼                               ▼                     ▼
   LOCAL (make)            RENTED GPU (cloud_setup.sh)      DOCKER (repro/serve)    SkyPilot (spot)
```

**Stage roadmap** (see `PLAN.md`): data → tokenizer → base pretrain → vision (toggle)
→ instruct (SFT) → reasoning (CoT/GRPO) → tools → eval → serve → quantize → apps.
**All stages are implemented and locally runnable at tiny scale (27 tests passing);**
remaining work is scaling up (real data volume + multi-GPU), not new components.

---

## 2. Tech stack — what & why (with alternatives)

| Layer | Tool used | Why | Alternatives | Trade-off |
|---|---|---|---|---|
| Core DL | **PyTorch** | De-facto research standard, dynamic, huge ecosystem | JAX/Flax, TensorFlow | JAX scales/compiles better (TPUs) but smaller ecosystem & steeper curve |
| Attention | **`F.scaled_dot_product_attention`** | Fused, uses FlashAttention kernels when available; zero extra deps | manual softmax(QKᵀ), `flash-attn` pkg | manual is clearer but slow/memory-heavy; flash-attn pkg is faster still but a fragile build |
| Tokenizer | **HF `tokenizers`** (byte-level BPE) + raw-byte mode | Fast Rust BPE; byte-level ⇒ no OOV | SentencePiece (unigram/BPE), tiktoken, pure bytes | SP is great for multilingual; tiktoken is fast but inference-only; pure bytes = 4× longer seqs |
| Config | **YAML + argparse** | Dead simple, readable, no magic | Hydra, OmegaConf, gin | Hydra composes/overrides better at scale; overkill for now |
| Experiment tracking | **TensorBoard** (pluggable to W&B) | Local, zero-account, fast | Weights & Biases, MLflow, Comet | W&B is better for teams/sweeps/artifacts; TB doesn't scale to 100s of runs |
| Packaging | **pyproject + `pip install -e .`** | Standard `src/` layout; editable for dev | Poetry, PDM, **uv** | uv is much faster & has lockfiles natively (likely next upgrade) |
| Reproducibility | **requirements.lock** | Pins exact versions | uv.lock, poetry.lock, conda env | lockfiles > editable install for "runs the same in 3 months" |
| Container | **Docker** (PyTorch CUDA base) | Pins CUDA/cuDNN/driver compat — kills the #1 cloud bug | bare venv, conda-pack, Singularity/Apptainer | Docker adds build time/size; needed for serving & K8s/Vertex |
| Launch/orchestration | **SkyPilot** (optional) | Cheapest spot GPU across clouds, managed auto-resume, 1 YAML | Kubeflow, Airflow, Flyte, Metaflow, raw VMs | Heavy orchestrators = a cluster to operate; unjustified for one team |
| Multi-GPU | **DDP** (built); FSDP/DeepSpeed ZeRO (scale path) | Data-parallel now; shard params/optimizer when model won't fit | Megatron-LM, TorchTitan | DDP can't fit huge models; FSDP/DeepSpeed for that; Megatron powerful but complex |
| Eval | **lm-evaluation-harness** (adapter built) + own benchmarks | Comparable numbers + fast local checks | HELM, custom | harness = community-trusted; ours = transparent/fast |
| Serving | **FastAPI** (built) / **vLLM** (production) | OpenAI-compatible API now; PagedAttention for throughput | TGI, SGLang, llama.cpp server | vLLM = best GPU throughput; llama.cpp wins on CPU/low-VRAM |
| Quantization | **int8 dynamic** (built); GGUF/AWQ/GPTQ/bitsandbytes | Shrink for small VRAM / CPU | — | trade quality vs size vs speed (measured per bit-width) |

### 2.1 Libraries (actual dependencies) — package · used for · alternatives

**Core (always installed — `requirements.txt` / `requirements.lock`):**

| Package | Used for in this project | Alternatives |
|---|---|---|
| `torch` | model, autograd, training loop, SDPA attention, DDP, int8 quant | JAX/Flax, TensorFlow |
| `tokenizers` (HF) | train/load the byte-level BPE tokenizer (`tokenizer/bpe.py`) | sentencepiece, tiktoken, raw bytes |
| `datasets` (HF) | stream FineWeb-Edu for pretraining + download benchmark data (`data/hf_stream.py`, `eval/tasks/`) | webdataset, mosaicml-streaming, custom |
| `numpy` | RAG vector store math, misc array ops | — |
| `pyyaml` | parse per-stage config files (`utils/config.py`) | tomli, json, Hydra/OmegaConf |
| `tqdm` | progress bars for long loops | rich, manual |
| `tensorboard` | local experiment tracking dashboard (`utils/metrics_logger.py`) | wandb, mlflow, comet |

**Optional (installed per feature — see `requirements.txt` notes):**

| Package | Used for | Alternatives |
|---|---|---|
| `lm-eval` | official benchmark numbers via the adapter (`eval/lm_eval_adapter.py`, `scripts/lm_eval_run.py`) | HELM, custom harness (our `benchmark.py`) |
| `transformers` | the pretrained **SigLIP** vision encoder toggle (`vision/encoder.py`) | open_clip, timm |
| `fastapi` + `uvicorn` + `pydantic` | OpenAI-compatible serving API (`serve/api.py`) | Flask, vLLM's own server, TGI |
| `accelerate` / `deepspeed` | scale DDP → FSDP/ZeRO for models too big for one GPU | raw `torch.distributed`, Megatron-LM, TorchTitan |
| `flash-attn` | faster/leaner attention kernels at scale (SDPA already uses them when present) | SDPA built-in, xformers |
| `trl` | drop-in SFT/DPO/GRPO if you prefer a framework over our transparent `align/` code | our own `train/`+`align/`, OpenRLHF, veRL |
| `vllm` | high-throughput production serving (PagedAttention) | TGI, SGLang, llama.cpp server |
| `bitsandbytes` | quick 4/8-bit GPU loading; GGUF (llama.cpp) / AWQ / GPTQ for deploy | — |
| `sentence-transformers` | stronger RAG embeddings than our hashing embedder | OpenAI/Cohere embeds (API), instructor |
| `faiss-cpu` / `chromadb` | scalable vector index for RAG (we ship a NumPy stand-in) | pgvector, Milvus, Qdrant |
| `wandb` | hosted/team experiment tracking (one-line swap from TensorBoard) | mlflow, comet, neptune |
| `skypilot` | launch cheapest spot GPU + managed auto-resume (`infra/sky/`) | Kubeflow, Airflow, Flyte, raw VMs |

Design intent: the **core** set is deliberately tiny so the whole pipeline runs
locally with no heavy deps; everything else is opt-in for scale/deploy. Our own
transparent modules (`train/`, `align/`, `eval/benchmarks.py`, `serve/api.py`,
`apps/rag/`) mean the project *works* without the optional libraries — they're
upgrades, not requirements.

---

## 3. Design decisions (the interview meat)

Each decision: **what, why, alternatives, pros/cons, best practice.**

### 3.1 Tokenizer: byte-level BPE default, raw-UTF-8 toggle
- **What:** default 32k byte-level BPE; a `byte` mode (256-vocab raw UTF-8) kept for comparison.
- **Why:** "UTF-8 vs BPE" is a false choice — *byte-level BPE is built on UTF-8 bytes plus learned merges*, so you get universality (no OOV, any language/emoji/code) **and** ~4× compression.
- **Pros (BPE):** ~4× shorter sequences ⇒ ~4× cheaper pretraining + 4× more effective context. **Cons:** tokenizer artifacts possible (glitch tokens, weak digit/math handling unless digits are split); needs a training step.
- **Pros (raw bytes):** zero training, never breaks, better at char-level tasks. **Cons:** 4× longer sequences ⇒ 4× cost; model must learn to compose bytes before reasoning.
- **Best practice:** reserve special tokens up front (chat/tool/vision) so every later stage shares one tokenizer; dedup + decontaminate corpus; consider digit-splitting for math.

### 3.2 Model architecture: modern Llama-style decoder
Hand-built but standard components — each is a defensible choice:
- **RMSNorm** vs LayerNorm → cheaper (no mean-centering), stable; standard in Llama/Mistral.
- **RoPE** vs learned/absolute pos-emb → relative position via rotation, extrapolates to longer ctx, no extra params.
- **SwiGLU** MLP vs GELU MLP → gated activation, better quality per parameter (costs a 3rd projection).
- **GQA** (fewer KV heads than Q heads) vs MHA → shrinks the KV cache ⇒ faster/cheaper inference; tiny quality hit. MQA (1 KV head) is the extreme.
- **Weight tying** (embedding = LM head) → saves `vocab×dim` params, regularizes.
- **SDPA** for attention → fused kernel, FlashAttention when available, causal masking for free.
- **Pros:** modern, efficient, transfers to real model sizes by only editing the config. **Cons:** hand-written ≠ HF-compatible yet, so TRL/vLLM/eval need a wrapper (planned next).
- **Best practice:** keep it size-agnostic via `ModelConfig`; prove correctness with a smoke-train (loss must drop, sample must memorize) before scaling.

### 3.3 Framework-first (not pure from-scratch)
- **Why:** the learning goal is the *whole pipeline*; reimplementing every lib wastes weeks. We hand-write the model (to understand it) but lean on HF/TRL/accelerate for plumbing.
- **Pros:** fast, battle-tested infra, interoperable. **Cons:** less "I built it all"; some abstraction to learn.
- **Alternative:** fully from scratch (max learning, slow) or fully framework (e.g. just call HF Trainer — least insight).

### 3.4 Vision = config toggle (from-scratch ViT ↔ pretrained SigLIP)
- **Why:** a *good* vision encoder needs hundreds of GPU-days — impossible at <$500/1 day. Toggle honors the from-scratch learning goal *and* the capability goal.
- **Pros:** honest about the trade-off; lets you A/B. **Cons:** from-scratch path will be weak at this budget (stated openly).
- **Best practice:** LLaVA-style 2-phase (train projector, then light LLM finetune); freeze the encoder when pretrained.

### 3.5 Pluggable logging abstraction
- **What:** a tiny `Logger` interface with TensorBoard/W&B/Noop backends; training code never imports a specific tool.
- **Why:** swap local↔hosted with one config line; testable (Noop).
- **Pros:** decoupled, future-proof. **Cons:** thin lowest-common-denominator API (advanced W&B features need direct calls).

### 3.6 Config-driven, CLI-per-stage
- **Why:** every stage is one entrypoint reading one YAML ⇒ reproducible, swappable, diffable.
- **Pros:** clean separation; scale model by editing numbers. **Cons:** YAML has no validation/typing (Hydra/pydantic would add it).

### 3.7 Reproducibility: lockfile + Docker + CUDA-pinned base
- **Why:** `pip install -e .` is a *dev convenience*, not reproducibility. The lockfile pins versions; the Docker base pins CUDA so there's no "CPU-only torch" surprise.
- **Pros:** identical env everywhere. **Cons:** image is GB-sized, build time.
- **Best practice:** stamp each run with git SHA + config hash + base-image tag (+ data hash).

### 3.8 Orchestration: SkyPilot, optional — not Kubeflow/Airflow
- **Why:** heavy orchestrators are a cluster you must operate; unjustified for a single-team training project. SkyPilot gives spot-GPU sourcing + auto-recovery from one YAML.
- **Pros:** production feel, minimal ops. **Cons:** another tool to install/configure; adopt only when manual pod-juggling hurts.

### 3.9 Spot instances + checkpointing (cost strategy)
- **Why:** spot GPUs are ~2–3× cheaper; pair with frequent checkpointing + auto-resume to survive preemption. Keeps the project under budget.
- **Cons:** jobs can be killed anytime ⇒ checkpointing is mandatory, not optional.
- **Resume is complete:** model + optimizer state + step + **data position** (`skip_blocks`) all restore, so a relaunched run continues through the corpus, not from the top. `--add-steps N` trains N more with a fresh warmup+cosine.

### 3.10 Line-ending normalization (`.gitattributes`)
- **Why:** dev-on-Windows → train-on-Linux; a CRLF in a `.sh` breaks bash (`bad interpreter: bash^M`). Forcing LF prevents a classic, hard-to-spot failure.

---

## 4. File-by-file map

| Path | What it does |
|---|---|
| `PLAN.md` | Full stage-by-stage roadmap + budget/scope reasoning |
| `ARCHITECTURE.md` | **This file** — design decisions, stack, run guide |
| `README.md` | Quickstart + the byte-vs-BPE rationale |
| **configs/** | |
| `configs/model_tiny.yaml` | Tiny knob-set for local smoke-train (model/tokenizer/data/train/logging) |
| **src/llmscratch/tokenizer/** | |
| `byte_tokenizer.py` | Raw UTF-8 byte tokenizer (vocab 256 + shared special tokens) |
| `bpe.py` | Byte-level BPE: train / save / load / encode / decode (HF `tokenizers`) |
| `__init__.py` | `build_tokenizer(cfg)` factory — picks `byte` or `bpe` |
| `compare.py` | CLI to *measure* UTF-8 vs BPE token counts on a sample |
| **src/llmscratch/model/** | |
| `decoder.py` | Llama-style decoder: RMSNorm, RoPE, SwiGLU, GQA, SDPA, generate(), `ModelConfig` |
| **src/llmscratch/data/** | |
| `text.py` | `iter_local_lines`, `encode_corpus`, `PackedDataset` (next-token windows) |
| `chat.py` | ChatML rendering + assistant-only loss masking; inference prompt builder |
| `hf_stream.py` | Streaming HF corpus (FineWeb-Edu) packed into blocks; `skip_blocks` resumes through the corpus (`datasets`) |
| **src/llmscratch/train/** | |
| `trainer.py` | Reusable loop: grad accum, cosine LR, clip, bf16, eval/ckpt hooks |
| **src/llmscratch/align/** | |
| `sft.py` | SFTDataset (loss-masked) + dynamic-pad collate + jsonl loader |
| `reasoning.py` | `<think>` CoT trace formatting + answer extraction |
| `grpo.py` | Group Relative Policy Optimization (verifiable-reward RL) |
| **src/llmscratch/tools/** | |
| `registry.py` | Tool/ToolRegistry + safe AST calculator |
| `parser.py` | Parse/execute `<tool_call>` JSON; tools system prompt |
| **src/llmscratch/eval/** | |
| `scoring.py` | `continuation_logprob` / `sequence_nll` (loglikelihood primitive) |
| `benchmarks.py` | Perplexity + length-normalized multiple-choice accuracy |
| `tasks/` | Real benchmark loaders+scorers: HellaSwag, OpenBookQA, GSM8K, BFCL, VQAv2 |
| `lm_eval_adapter.py` | EleutherAI lm-eval-harness adapter (`score_pair` + `LMScratchLM`) |
| **src/llmscratch/serve/** | |
| `generate.py` | Sampling (top-k/top-p, stop tokens) + `generate_chat` |
| `api.py` | OpenAI-compatible FastAPI (`/v1/chat/completions`); vLLM-contract |
| **src/llmscratch/quantize/** | |
| `dynamic_int8.py` | Dynamic int8 quantization + serialized-size measurement |
| **src/llmscratch/vision/** | |
| `vit.py` | From-scratch Vision Transformer (patch embed + encoder) |
| `encoder.py` | **Toggle**: from-scratch ViT ↔ frozen pretrained SigLIP |
| `projector.py` | MLP vision→LLM dim (LLaVA phase-1 trainable) |
| `multimodal.py` | Splices projected patches into `<image>` positions |
| `data.py` | Synthetic image→caption dataset + collate (trainable offline) |
| **src/llmscratch/apps/** | |
| `rag/` | Hashing embedder + cosine VectorStore + grounded RAG pipeline |
| `agent/` | Model-agnostic ReAct loop driving the tool registry |
| **src/llmscratch/utils/** | |
| `metrics_logger.py` | Pluggable logger: TensorBoard / W&B / Noop + `build_logger` |
| `config.py` | Config load + run provenance (`run_id` = gitSHA+cfgHash) |
| `checkpoint.py` | Atomic save/load + `find_latest` for spot-safe resume |
| `distributed.py` | torchrun-aware DDP setup + is_main gating (NCCL/gloo) |
| **scripts/** | |
| `run_all.py` | **Orchestrator** — runs every stage sequentially (`--smoke` or ~2h local) |
| `demo.py` | One-command local check — runs every stage on CPU in seconds |
| `train_tokenizer.py` | Train the byte-level BPE tokenizer from a config |
| `smoke_train.py` | Minimal end-to-end train loop (cosine LR, logging, sampling) |
| `pretrain.py` | Base pretraining — single-GPU or DDP via `torchrun`; auto-resume |
| `sft.py` | Instruct/tool SFT from a base checkpoint |
| `train_vision.py` | Multimodal LLaVA training (phase 1 / phase 2) |
| `evaluate.py` | Quick local eval: perplexity + custom MCQ smoke set |
| `benchmark.py` | Real benchmarks (HellaSwag/OpenBookQA/GSM8K/BFCL) — local, no API |
| `lm_eval_run.py` | Run EleutherAI lm-eval-harness (official, comparable numbers) |
| `quantize.py` | Quantize a checkpoint; report size + perplexity delta |
| `check_ddp.py` | Verify the DDP path locally (2 ranks, CPU, FileStore) |
| **tests/** (27 passing) | |
| `test_tokenizer.py` | Round-trip + compression + special-token tests |
| `test_model.py` | Forward shapes, GQA divisibility, generation, **causality** |
| `test_tools.py` `test_eval.py` | Tool calculator/parser; perplexity/MCQ scoring |
| `test_apps.py` `test_vision.py` `test_reasoning.py` | RAG/agent; multimodal; CoT/GRPO |
| **infra/** | |
| `infra/cloud_setup.sh` | No-Docker fast path: pinned install + GPU sanity + smoke-train |
| `infra/sky/train.yaml` | SkyPilot launcher (cheapest spot GPU, managed auto-resume) |
| `infra/README.md` | When to use local vs rented-box vs Docker; orchestrator advice |
| **root infra** | |
| `Dockerfile` | CUDA-correct image (PyTorch base, cached dep layer, sanity CMD) |
| `.dockerignore` | Keeps build context/image lean |
| `requirements.txt` | Loose deps for local dev |
| `requirements.lock` | Pinned deps for reproducible cloud runs |
| `pyproject.toml` | Package metadata, `src/` layout, pytest config |
| `Makefile` | One-word commands (setup/smoke/tb/test/docker-*/sky-*) |
| `.gitignore` | Excludes runs/artifacts/checkpoints/secrets |
| `.gitattributes` | Forces LF endings (Windows→Linux safety) |
| `data/sample.txt` | Tiny corpus for offline smoke-train |

---

## 5. How to run

### Local (quick tests — Windows or any OS)
```bash
pip install -r requirements.txt
pip install -e .

# verify the whole pipeline (no GPU / no downloads):
python scripts/demo.py        # SEE every stage run in seconds
pytest                        # RUN 27 invariant tests

# training stages (tiny, CPU-friendly):
python scripts/pretrain.py --config configs/pretrain_tiny.yaml       # base + checkpoints
python scripts/sft.py      --config configs/sft_tiny.yaml            # instruct
python scripts/train_vision.py --phase 2                            # multimodal
python scripts/evaluate.py --ckpt artifacts/ckpt_pretrain/step_0000200.pt
python scripts/quantize.py --ckpt artifacts/ckpt_pretrain/step_0000200.pt
tensorboard --logdir runs                                           # training curves
```
On Linux/macOS/git-bash, shortcuts: `make setup`, `make smoke`, `make tb`, `make test`.

**Multi-GPU** (Linux + NVIDIA): the same script scales via `torchrun` —
```bash
torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain_300m.yaml
```
(For models too big for one GPU, switch DDP→FSDP/DeepSpeed via `accelerate config`;
the loop is unchanged. Verify the DDP path locally with `python scripts/check_ddp.py`.)

### Cloud — pick one path
**A. Rented GPU box, no Docker (fastest start):** spin up a RunPod/Lambda/vast.ai pod
on a **PyTorch+CUDA** template, then:
```bash
git clone <repo> && cd llm-from-scratch
bash infra/cloud_setup.sh        # pinned install + GPU check + smoke-train
```
**B. Docker (reproducible / serving):**
```bash
make docker-build                # docker build -t llm-from-scratch:dev .
make docker-run                  # docker run --rm --gpus all ...
```
**C. SkyPilot (spot GPU, auto-resume):**
```bash
pip install "skypilot[aws,gcp,runpod]"
sky jobs launch -n llm-train infra/sky/train.yaml
sky down llm
```

---

## 6. Updating common parameters

All knobs live in `configs/*.yaml` — no code edits needed.

| Goal | Where | Change |
|---|---|---|
| **Bigger/smaller model** | `model:` | `dim`, `n_layers`, `n_heads`, `n_kv_heads` (n_heads % n_kv_heads == 0), `max_seq_len` |
| **Switch tokenizer** | `tokenizer.mode` | `bpe` ↔ `byte` (byte needs no training step) |
| **Tokenizer vocab** | `tokenizer.vocab_size` | e.g. 32000 for real runs |
| **Train longer / LR** | `train:` | `steps`, `lr`, `warmup_steps`, `weight_decay`, `batch_size` |
| **Context / packing** | `data.block_size` | must be ≤ `model.max_seq_len` |
| **Sampling frequency** | `train.sample_every` | how often a generation is logged |
| **Logging backend** | `logging.backend` | `tensorboard` ↔ `wandb` ↔ `none` (+ `project`, `run_name`) |
| **CPU vs GPU** | `train.device` | `auto` / `cuda` / `cpu` |
| **Number of GPUs** | launch cmd | `torchrun --nproc_per_node=N` (effective batch ×= N) |
| **Grad accumulation** | `train.grad_accum` | effective batch = batch_size × grad_accum × N_gpus |
| **Cloud GPU count/type** | `infra/sky/train.yaml` | `resources.accelerators`, `use_spot` |
| **Docker CUDA version** | `Dockerfile` | base image tag (must be ≤ host driver's CUDA) |

**Scaling-up recipe:** copy `model_tiny.yaml` → `model_300m.yaml`, bump `dim≈1024`,
`n_layers≈24`, `max_seq_len=2048`, `vocab_size=32000`, point `data.corpus` at the HF
streaming loader (next stage), set `logging.backend: wandb`, launch via SkyPilot on
8×A100. Nothing else changes — that's the payoff of config-driven design.

---

## 7. Best practices baked in
- **Config-driven everything** → reproducible, swappable, diffable.
- **Smoke-train before scaling** → prove the loop on a tiny corpus (loss drops, model memorizes) before spending on GPUs.
- **Pin deps + pin CUDA** → reproducibility comes from lockfile + base image, not `-e .`.
- **Spot + checkpoint + auto-resume** → cheap and preemption-safe.
- **Decouple from vendors** (logger abstraction, toggleable tokenizer/vision) → swap tools without touching training code.
- **Test invariants, not outputs** → shapes, GQA divisibility, and *causality* (future tokens can't change past logits).
- **Stamp runs** with git SHA + config + image tag for traceability.

---

## 8. Likely interview questions (talking points)
- *Why byte-level BPE over raw UTF-8?* → false dichotomy; BPE is built on bytes; 4× compression ⇒ 4× cheaper + more context.
- *Why GQA?* → smaller KV cache ⇒ cheaper/faster inference at near-MHA quality.
- *Why RoPE over learned positions?* → relative, extrapolates, parameter-free.
- *How do you keep training reproducible?* → lockfile + CUDA-pinned Docker + run stamping; `-e .` is just dev ergonomics.
- *Do you need Kubernetes?* → no; SkyPilot covers spot + auto-resume without operating a cluster. K8s/Kubeflow only at multi-team scale.
- *How would you scale this 300M→7B?* → FSDP/DeepSpeed ZeRO-3 sharding, gradient checkpointing, sequence/tensor parallel, more nodes; config + launcher already abstract it.
- *How do you survive spot preemption?* → checkpoint every N steps to object storage + `sky jobs launch` auto-recovery.
- *Why not a vision encoder fully from scratch for quality?* → compute-prohibitive; toggle to SigLIP and only train the projector.
- *How do you validate the model isn't just memorizing position?* → causality test + held-out eval via lm-eval-harness with decontamination.
