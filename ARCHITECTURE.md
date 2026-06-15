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
   (local .txt / .bin /                             ▼                          │ scalars/text
    HF-stream / mixing)                       ┌──────────┐  loss/logits        │
                                              │  MODEL   │─────────────────────┘
                                              │ decoder  │  (RoPE/RMSNorm/
                                              │  .py     │   SwiGLU/GQA)
                                              └────┬─────┘
                                                   │ AdamW + cosine LR
                                        ┌──────────▼───────────┐
                                        │   TRAIN LOOP          │  Trainer (DDP, resume,
                                        │  pretrain→SFT→reason  │  time-box, compile)
                                        │  →eval→quant→export   │  → checkpoints + samples
                                        └──────────┬───────────┘
                                                   │
        ┌──────────────────────────┬──────────────┴───────────────┬─────────────────────┐
        ▼                          ▼                               ▼                     ▼
   LOCAL (run_all)        RENTED GPU (cloud_setup.sh)      DOCKER (repro/serve)    SkyPilot (spot)
```

**Stage roadmap** (see `PLAN.md`): data → tokenizer → base pretrain → vision (toggle)
→ instruct (SFT) → reasoning (CoT/GRPO) → tools → eval → serve → quantize → **HF/GGUF
export** → apps. **All stages implemented and runnable (47 tests).** `scripts/run_all.py`
chains them (see §3B); checkpoints export to a real HF Llama / GGUF (see §3C). Remaining
gaps are lower-priority (FSDP/DeepSpeed, DPO, contrastive ViT, AWQ/GPTQ, robust safety).

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
- **Pros:** modern, efficient, transfers to real model sizes by only editing the config. The
  arch uses HF's `rotate_half` RoPE, so it **exports losslessly to `transformers.LlamaForCausalLM`**
  (§3C) → TRL/vLLM/lm-eval/GGUF for free. **Cons:** hand-written core to maintain.
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

## 3A. Datasets & data formats per stage

What data each stage consumes, the **on-disk/in-memory format**, and **how it becomes
a training signal**. Two mechanics recur and are worth fixing in your head first:

- **Packing (pretraining):** tokenized documents are concatenated into one long stream
  and sliced into fixed windows of `block_size+1` tokens. For a window `w`, the model
  input is `x = w[:-1]` and the target is `y = w[1:]` — i.e. *predict the next token at
  every position*. Loss = cross-entropy(logits, y) over all positions. No labels needed
  → **self-supervised**.
- **Loss masking (post-training):** for chat/instruct/tool/reasoning data we set target
  tokens we don't want to learn to `-100` (PyTorch `ignore_index`). So the model is
  trained to *produce* assistant turns but not to reproduce the user's text or tool
  outputs. Same next-token objective, just masked.

| Stage | Dataset (example) | Format | How it trains the model |
|---|---|---|---|
| **Tokenizer** | FineWeb-Edu sample (stream) | raw text docs (`{"text": "..."}`) | feed raw strings to the byte-level BPE trainer to learn merges; **no model, no labels** — just vocabulary |
| **Base pretrain** | `HuggingFaceFW/fineweb-edu` (sample-10BT), streamed | `{"text": "<web document>"}` | tokenize → `[BOS] doc [EOS] [BOS] doc …` → pack into `block_size` windows → next-token prediction over every token |
| **Instruct SFT** | UltraChat / OpenHermes / Tulu subset (→ jsonl) | `{"messages": [{"role","content"}, …]}` | render ChatML with special tokens; **mask all but assistant tokens**; next-token loss on the assistant reply |
| **Tool-use SFT** | xLAM / Hermes-function-calling (our `sample_tools_chat.jsonl`) | messages where assistant emits `<tool_call>{json}</tool_call>` and a `tool` role returns the result | same as SFT; assistant turns (incl. the tool call) are unmasked, `user`+`tool` turns masked → learns *when/how to call* + how to answer from the result |
| **Reasoning — CoT distill** | GSM8K (via `prepare_reason.py`) / OpenR1 traces | assistant content = `<think>{reasoning}</think>{answer}` | folded into the **merged SFT pass** (with instruct/tools/safety) so it learns to reason without forgetting chat; loss on the whole assistant turn |
| **Reasoning — GRPO (RL)** | math prompts with **verifiable** answers (GSM8K-style) | `{question, gold_answer}` — *no target completion* | sample G completions per prompt → reward = is-answer-correct → group-normalized advantage → policy-gradient update (online, no labels to imitate) |
| **Multimodal** | LAION/COCO/LLaVA mixture (our synthetic color set offline) | `{image: tensor[3,H,W], text}`; prompt holds `<image>` placeholder tokens | image → encoder(ViT/SigLIP) → projector → embeddings **spliced into the `<image>` positions**; next-token loss on the caption/answer (assistant-masked). Phase 1 trains projector only; phase 2 + LLM |
| **Evaluation** | HellaSwag, OpenBookQA, GSM8K, BFCL, VQAv2 | see below | **no training** — measurement only |

**Real-data prep (one-time downloads):** `prepare_instruct.py` (UltraChat → `data/instruct.jsonl`,
auto-detects messages/ShareGPT schemas), `prepare_tools.py` (xLAM → `<tool_call>` jsonl),
`prepare_reason.py` (GSM8K CoT → `data/reason.jsonl`). `run_all` uses these if present,
else tiny offline placeholders. **Mixing:** `prepare_data.py --datasets` weight-interleaves
corpora (e.g. `--minipile-local` pretrains on 90% MiniPile + 10% `the-stack-smol` code).

**Streaming vs. offline (pretrain):** by default the corpus is **streamed**
(`source: hf`, `streaming=True`) — TB-scale data is fetched shard-by-shard *during*
training, not downloaded upfront (hence occasional CDN retry warnings). `prepare_data.py`
+ `source: bin` pre-tokenizes a fixed sample to a local uint16 file for a fully offline
run (random windows, no network, trivial resume).

**Concrete shapes**

- *Pretrain window* (`block_size=4` toy): tokens `[the, cat, sat, on, mat]` →
  `x=[the,cat,sat,on]`, `y=[cat,sat,on,mat]`.
- *SFT (ChatML, masked)*: `<bos> <im_start>user\nHi<im_end> <im_start>assistant\nHello<im_end> <eos>`
  — labels are `-100` everywhere except the `assistant\nHello<im_end>` span, so only the
  reply contributes to the loss. (`data/chat.py:render_chat`)
- *Multimodal*: `input_ids = [BOS, <image>×N, "what color?", …, assistant, "a red image", EOS]`;
  the `N` `<image>` rows of the embedding matrix are overwritten by the projected patch
  embeddings before the decoder runs. (`vision/multimodal.py`)

**Eval data formats** (all scored locally — `eval/tasks/`):

| Benchmark | Format | Scoring |
|---|---|---|
| HellaSwag / OpenBookQA | `{question, choices[], answer:int}` | pick the choice with highest length-normalized log-likelihood (`acc_norm`) |
| GSM8K | `{question, answer:"… #### 42"}` | few-shot CoT generate → extract final number → exact match |
| BFCL | `{question, tools[], gold:{name,arguments}}` | parse emitted `<tool_call>` → AST/argument match vs gold |
| VQAv2 | `{image, question, answers:[10 strings]}` | generate → VQA soft-accuracy `min(matches/3, 1)` |

---

## 3B. The `run_all` orchestrator & profiles

`scripts/run_all.py` runs the whole lifecycle **sequentially as subprocesses** (so each
stage's output streams live and the pyarrow-before-torch import order is clean per
process). It is **resumable** (re-run = continue: pretrain auto-loads model + optimizer
+ data position) and **idempotent** for prep (skips downloads/tokenizers that already
exist).

**Stage order:**
```
tokenizer → [1b prep pretrain .bin] → [1c prep REAL post-train data]
   → pretrain (resumable, time-boxed or step-count; VRAM preflight + bin-size warning)
   → base eval (stage 3)
   → SFT — ONE merged pass: instruct + tool-use + safety + reasoning(CoT) → no forgetting
   → final eval (perplexity+MCQ; + real benchmarks for real-data profiles)
   → quantize → sample (chats; emits <think>…</think> when reasoning)
```

**Profiles** (`--flag`), each a self-contained config/ckpt set so they never collide:

| Profile | Model | Pretrain data | Post-train data | GPU | Network |
|---|---|---|---|---|---|
| `--smoke` | ~1M | local `sample.txt` | placeholders | CPU | none (wiring check, ~1 min) |
| *(default)* | ~25M | FineWeb-Edu **stream** | placeholders | 6 GB | yes (streams) |
| `--offline` | ~25M | FineWeb-Edu local `.bin` | placeholders | 6 GB | once (prep) |
| `--minipile-local` | ~41M | **MiniPile + code** `.bin` | **real** (UltraChat/xLAM/GSM8K) | 6 GB | once (prep) |
| `--minipile` | ~1B | MiniPile `.bin` | real | 40–80 GB | once (prep) |

**Key design points**
- **Merged SFT mix (incl. reasoning):** instruct + tool-use + safety + reasoning-CoT train
  in *one* shuffled SFT pass, not chained passes. **Why it matters:** a separate
  reasoning pass on GSM8K-only data made the model answer *every* prompt with math
  (catastrophic forgetting) — training all four together keeps general chat *and* teaches
  `<think>` reasoning. (GRPO RL remains an optional separate stage.) Each component
  auto-uses real prepared data (`prepare_instruct/tools/reason`) if present, else a tiny
  offline placeholder.
- **Real vs placeholder:** the `prepare_*` scripts fetch real data once → `data/*.jsonl`;
  `run_all` detects and uses them. Keeps `--smoke`/`--offline` fully offline.
- **Final vs base eval:** the model is eval'd after pretraining *and* after all
  post-training, so you can read what each stage bought.
- **Guardrails (fail fast, not silently):** pretrain runs a **VRAM preflight** (aborts
  with a clear message if `params×16B` won't fit — on Windows an oversize model otherwise
  spills to shared RAM and *looks stuck*); a **bin-size/epochs warning** (small corpus ⇒
  memorization: low train loss, huge held-out perplexity); and `run_all` **fails stages
  cleanly** (exit code + one-line stop, no buried traceback).
- **Time-box vs steps:** desktop profiles stop on `--pretrain-minutes`; cloud profiles
  run the config's `steps`. `--add-steps N` extends a finished model with a fresh schedule.

## 3C. Export & deployment

The transparent path (our `serve/`, `quantize/`) is for understanding; the **export
path** turns a checkpoint into standard artifacts for real serving:

- **HF Llama export** (`model/hf_export.py`): maps our Decoder onto a genuine
  `transformers.LlamaForCausalLM`. Our arch already uses HF's **`rotate_half` RoPE**
  convention, so q/k weights transfer with **no permutation** — verified **numerically
  identical** (max logit diff ~5e-7). This unlocks TRL, **vLLM**, and native lm-eval.
- **GGUF** (`scripts/export_gguf.py`): HF folder → llama.cpp `convert_hf_to_gguf.py` →
  `llama-quantize` (e.g. Q4_K_M). The "runs anywhere" path: CPU / Apple Silicon /
  low-VRAM (Ollama, LM Studio).
- **Serving:** `serve/api.py` is an OpenAI-compatible FastAPI over our model (transparent);
  for throughput, export to HF and `vllm serve` (PagedAttention, same API contract).
- **Quantization ladder:** int8 dynamic (built, CPU) → GGUF/llama.cpp (laptop) → AWQ/GPTQ
  4-bit (GPU serving) → bitsandbytes (quick tests). Always report quality (perplexity)
  next to the size/speed win.

---

## 4. File-by-file map

| Path | What it does |
|---|---|
| `PLAN.md` | Full stage-by-stage roadmap + budget/scope reasoning |
| `ARCHITECTURE.md` | **This file** — design decisions, stack, run guide |
| `README.md` | Quickstart + the byte-vs-BPE rationale |
| `RUNBOOK.md` | Step-by-step cloud training run (provision → … → serve, cost controls) |
| `CHEATSHEET.md` | One-page interview-prep summary |
| `READINESS.md` | End-user readiness: safety, prompt-injection, tool sandboxing, licensing |
| `MODEL_CARD.md` | Generated by `scripts/model_card.py` (arch, data, eval, limits, license) |
| **configs/** (one per stage/profile) | |
| `model_tiny.yaml` · `pretrain_{tiny,local,local_offline,300m,minipile,minipile_local}.yaml` | model + pretrain knob-sets (size, data source, schedule) |
| `tokenizer_{32k,local,minipile}.yaml` · `sft_{tiny,local,tools_tiny,minipile,minipile_local}.yaml` | tokenizer training + SFT knob-sets |
| **src/llmscratch/tokenizer/** | |
| `byte_tokenizer.py` | Raw UTF-8 byte tokenizer (vocab 256 + shared special tokens) |
| `bpe.py` | Byte-level BPE: train / save / load / encode / decode (HF `tokenizers`) |
| `__init__.py` | `build_tokenizer(cfg)` factory — picks `byte` or `bpe` |
| `compare.py` | CLI to *measure* UTF-8 vs BPE token counts on a sample |
| **src/llmscratch/model/** | |
| `decoder.py` | Llama-style decoder: RMSNorm, RoPE, SwiGLU, GQA, SDPA, generate(), `ModelConfig` |
| `hf_export.py` | Export our Decoder → genuine `transformers.LlamaForCausalLM` (numerically exact; unlocks TRL/vLLM/lm-eval/GGUF) |
| **src/llmscratch/data/** | |
| `text.py` | `iter_local_lines`, `encode_corpus`, `PackedDataset` (next-token windows) |
| `chat.py` | ChatML rendering + assistant-only loss masking; inference prompt builder |
| `hf_stream.py` | Streaming HF corpus (FineWeb-Edu) packed into blocks; `skip_blocks` resumes through the corpus (`datasets`) |
| `bin_data.py` | Train from a pre-tokenized local `.bin` (offline, random windows, no network) |
| `mixing.py` | `weighted_interleave` — mix multiple corpora (e.g. web text + code) by weight |
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
| `safety.py` | Refusal detection + refusal-rate / over-refusal safety report |
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
| `prepare_data.py` | Pre-download + tokenize a sample to a local `.bin` (offline training); `--datasets` mixes corpora |
| `prepare_instruct.py` | Fetch a real instruct dataset (UltraChat/OpenHermes) → `data/instruct.jsonl` |
| `prepare_tools.py` | Fetch a real function-calling dataset (xLAM) → `data/tools.jsonl` |
| `prepare_reason.py` | Fetch real CoT (GSM8K) → `data/reason.jsonl` |
| `smoke_train.py` | Minimal end-to-end train loop (cosine LR, logging, sampling) |
| `pretrain.py` | Base pretraining — single-GPU or DDP via `torchrun`; auto-resume |
| `sft.py` | Instruct/tool SFT from a base checkpoint |
| `train_vision.py` | Multimodal LLaVA training (phase 1 / phase 2) |
| `evaluate.py` | Quick local eval: perplexity + custom MCQ smoke set |
| `benchmark.py` | Real benchmarks (HellaSwag/OpenBookQA/GSM8K/BFCL) — local, no API |
| `lm_eval_run.py` | Run EleutherAI lm-eval-harness (official, comparable numbers) |
| `quantize.py` | Quantize a checkpoint; report size + perplexity delta |
| `export_hf.py` | Export a checkpoint to a HF Llama folder (TRL / vLLM / lm-eval) |
| `export_gguf.py` | Export to GGUF via llama.cpp (CPU / low-VRAM, "runs anywhere") |
| `model_card.py` | Generate MODEL_CARD.md (arch, data, eval, limitations, license) |
| `chat.py` | **Prompt your own model** — interactive REPL or `--prompt`; chat/complete modes |
| `status.py` | Check live loss/throughput from `metrics.jsonl` (no browser); `--watch` |
| `check_ddp.py` | Verify the DDP path locally (2 ranks, CPU, FileStore) |
| **tests/** (47 passing) | |
| `test_tokenizer.py` | Round-trip + compression + special-token tests |
| `test_model.py` | Forward shapes, GQA divisibility, generation, **causality** |
| `test_tools.py` `test_eval.py` | Tool calculator/parser; perplexity/MCQ scoring |
| `test_apps.py` `test_vision.py` `test_reasoning.py` | RAG/agent; multimodal; CoT/GRPO |
| `test_hf_export.py` | **Decoder == exported Llama** logits (numerically exact) |
| `test_data_resume.py` | data-position skip + `weighted_interleave` mixing + bin windows |
| `test_benchmark_tasks.py` `test_safety.py` `test_logger.py` | GSM8K/BFCL/VQA scoring; refusal eval; logger fallback |
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
| `data/sample_{chat,tools_chat,safety,reason,mcq,bfcl}.jsonl` | Tiny offline placeholders (instruct/tools/safety/CoT) + eval fixtures |

---

## 5. How to run

### Local (quick tests — Windows or any OS)
```bash
pip install -r requirements.txt
pip install -e .

# verify the whole pipeline (no GPU / no downloads):
python scripts/demo.py        # SEE every stage run in seconds
pytest                        # RUN 47 invariant tests

# run the WHOLE pipeline via the orchestrator (see §3B for profiles):
python scripts/run_all.py --smoke           # tiny, offline, ~1 min (wiring check)
python scripts/run_all.py --minipile-local  # ~41M, real data (MiniPile+code, UltraChat/xLAM/GSM8K), final eval

# or individual stages (tiny, CPU-friendly):
python scripts/pretrain.py --config configs/pretrain_tiny.yaml       # base + checkpoints
python scripts/sft.py      --config configs/sft_tiny.yaml            # instruct
python scripts/train_vision.py --phase 2                            # multimodal
python scripts/chat.py --ckpt <ckpt> --tokenizer <tok.json>         # prompt your model
python scripts/export_hf.py --ckpt <ckpt> --out artifacts/hf_model  # → HF Llama (vLLM/TRL/GGUF)
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
| **torch.compile** | `train.compile` (+`compile_mode`) | `true` for Linux-GPU speedup (often 1.3-2×); auto-skipped on CPU/Windows. `reduce-overhead` uses CUDA graphs (+VRAM) |
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
- *You hand-wrote the model — how do you use vLLM/TRL/GGUF then?* → export to a real `transformers.LlamaForCausalLM`; our RoPE matches HF `rotate_half`, so weights map with no permutation — verified logit-identical (~5e-7). From there vLLM/TRL/lm-eval/llama.cpp all work.
- *How do you add tool-use/reasoning without forgetting chat?* → train one *merged* SFT mixture (instruct + tools + safety + CoT), not chained passes. War story: a separate GSM8K-only reasoning pass made a small model answer *every* prompt with math (catastrophic forgetting) — merging fixed it. Small models forget the previous task fast; keep the data balanced in one pass.
- *Run looks stuck / model gives garbage?* → two guards: a **VRAM preflight** (oversize model on Windows spills to shared RAM and crawls, not OOM) and a **bin-size warning** (small corpus ⇒ memorization: ~0 train loss but ~1e6 held-out perplexity).
- *Real or toy datasets?* → `prepare_{instruct,tools,reason}.py` fetch UltraChat / xLAM / GSM8K; `--minipile-local` runs the full real pipeline (web+code pretrain → real SFT → final benchmarks); placeholders keep offline runs working.
