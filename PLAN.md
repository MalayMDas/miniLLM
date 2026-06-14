# LLM-from-Scratch — Project Plan

A modular, learning-oriented pipeline to build, train, evaluate, deploy, quantize,
and *use* a small multimodal LLM. **Framework-first** (PyTorch + HuggingFace
ecosystem), targeting **< $500** on rented multi-GPU, with **per-stage runs that fit
within ~a day**.

> **Build status:** every stage below is implemented and runs locally at tiny scale
> (27 tests passing). What remains is *scaling up* — real data volume, multi-GPU
> (FSDP/DeepSpeed), and the actual cloud training run — not new components. See
> `ARCHITECTURE.md` for the file-by-file map.

---

## 0. Reality check & headline decisions

Your answers had one unavoidable tension. Being honest up front so we design around it:

| Your choice | Reality | How the plan handles it |
|---|---|---|
| Vision encoder **from scratch** | A *good* encoder (CLIP/SigLIP) needs ~400M+ pairs & hundreds of GPU-days. Impossible in 1 day / <$500. | Vision is a **config toggle**: `from_scratch` ViT (default, for learning) vs `pretrained` SigLIP (for capability). You pick per run. |
| **As capable as budget allows** | <$500 + 1 day caps us at a **~300M-param** text model + small ViT. | We pick a Llama-style arch that's small but *modern* (RoPE, SwiGLU, RMSNorm, GQA) so it punches above its size. |
| **Within a day** | Whole pipeline ≠ 1 day. *Each stage's training run* can fit in hours. | Stages are independent & checkpointed; you run them sequentially over a few sessions, each < 1 day. |
| **Framework-first** | Good — saves weeks. | Define our model as an HF-compatible `PreTrainedModel` so TRL, vLLM, lm-eval-harness, and quant tools all work for free. |

**Target model spec (the "capable within budget" sweet spot):**
- Text decoder: **~300M params** (Llama-style: 24 layers, d=1024, 16 heads, GQA, RoPE, SwiGLU, RMSNorm), ctx 2048.
- Tokenizer: **our own BPE**, 32k vocab, trained on a data sample.
- Vision (toggle): from-scratch **ViT ~60M** *or* frozen **SigLIP-so400m**; LLaVA-style projector.
- Pretraining tokens: **~10–20B** (FineWeb-Edu sample) — Chinchilla-ish for this size.

**Compute recommendation:** 8×A100-40GB (or 4×H100) for ~a day.
- **Cheapest:** Lambda Cloud / RunPod / vast.ai (~$8–14/hr for 8×A100) → a full day ≈ $200–350.
- **GCP (you mentioned it):** `a2-highgpu-8g` (8×A100) is pricier (~$30/hr on-demand, ~$10–12 Spot). Use **Spot/preemptible** + checkpointing to stay < $500. Plan includes GCP IaC either way.

> If budget is the hard wall, we can drop to a **single A100 + ~124M model** (GPT-2-class, ~$30–60 for the base run) and still do every stage. The code is size-agnostic via config.

---

## 1. Repository structure (modular by design)

```
llm-from-scratch/
├── configs/                 # YAML configs — one per stage, swap sizes/datasets here
│   ├── model_300m.yaml
│   ├── tokenizer.yaml
│   ├── pretrain.yaml
│   ├── vision.yaml          # toggle: from_scratch | pretrained
│   ├── sft.yaml  rl.yaml  eval.yaml  serve.yaml  quant.yaml
├── src/llmscratch/
│   ├── data/                # HF datasets streaming, mixing, packing, tokenization
│   ├── tokenizer/           # train + load BPE (HF tokenizers)
│   ├── model/               # decoder, attention, RoPE, MoE-off, HF wrapper
│   ├── vision/              # ViT-from-scratch, SigLIP loader, projector, fusion
│   ├── train/               # pretrain loop (accelerate+DeepSpeed ZeRO), callbacks
│   ├── align/               # SFT, DPO, GRPO (via TRL) for instruct+reasoning
│   ├── tools/               # tool-use schema, function-calling data formatting
│   ├── eval/                # lm-eval-harness wiring + custom benchmarks
│   ├── serve/               # vLLM server, OpenAI-compatible API
│   ├── quantize/            # GGUF/llama.cpp, AWQ, GPTQ, bitsandbytes
│   └── apps/                # rag/  toolagent/  agent/  (demos)
├── infra/gcp/               # Terraform + startup scripts, Spot VMs, GCS buckets
├── scripts/                 # one-command entrypoints per stage
├── notebooks/               # exploration + visualization per stage
└── tests/                   # unit tests on shapes, masking, data, tokenizer
```

**Best practice:** every stage = a CLI entrypoint reading a YAML config, writing a
checkpoint + a run card (metrics, data hash, config) to GCS. Reproducible, resumable.

---

## 2. Stage-by-stage plan

### Stage 1 — Data (free, from HuggingFace)
- **Pretrain text:** `HuggingFaceFW/fineweb-edu` (sample-10BT/100BT) — high quality, the
  biggest lever for a small model. Mix a little code: `bigcode/the-stack-smol`.
- **Vision pretrain pairs:** `conceptual_captions` / `laion/laion-coco` subset, plus
  `COCO captions` for alignment eval.
- **Instruct/SFT:** `HuggingFaceH4/ultrachat_200k`, `teknium/OpenHermes-2.5`,
  `allenai/tulu-3-sft-mixture` (pick a subset).
- **Reasoning:** `open-r1/OpenR1-Math` / `nvidia/OpenMathReasoning` (CoT traces),
  distilled from a strong open teacher.
- **Tool use:** `Salesforce/xlam-function-calling-60k`, `NousResearch/hermes-function-calling`.
- **Module:** streaming loader, dataset mixing weights, sequence packing, on-the-fly
  tokenization, caching to GCS.
- *Pros:* free, large, well-documented. *Cons:* licenses vary (we log per-dataset
  license); quality filtering matters. *Best practice:* dedup + decontaminate vs eval sets.

### Stage 2 — Tokenizer (from scratch)
- Train a **32k BPE** (`tokenizers` lib) on a FineWeb-Edu sample; byte-fallback; special
  tokens for chat/tool/vision (`<|im_start|>`, `<tool_call>`, `<image>`).
- *Best practice:* reserve special tokens now so every later stage shares one tokenizer.

### Stage 3 — Base model pretraining (text)
- Custom Llama-style decoder as an **HF `PreTrainedModel`** (gives us free TRL/vLLM/eval).
- Training: **`accelerate` + DeepSpeed ZeRO-2/3**, bf16, FlashAttention-2, grad checkpointing,
  cosine LR + warmup, ~10–20B tokens. WandB logging, GCS checkpointing, auto-resume (for Spot).
- *Outcome:* a base model that completes text coherently. ~½ day on 8×A100.

### Stage 4 — Multimodal (vision, the toggle)
- **From-scratch path (default):** small ViT trained contrastively (CLIP-style) on
  image-text pairs → then LLaVA-style 2-phase: (a) train projector only, (b) finetune
  projector+LLM on multimodal instructions.
- **Pretrained path (capable):** freeze SigLIP, train only the projector (+light LLM FT).
- *Honest note:* from-scratch vision will be weak at this budget; toggle lets you compare.

### Stage 5 — Instruct model (SFT)
- Chat-format SFT via **TRL `SFTTrainer`** on UltraChat/Hermes/Tulu subset; ChatML template.
- LoRA optional for cheap iteration; full FT if budget allows.

### Stage 6 — Reasoning model
- **(a) CoT distillation:** SFT on reasoning traces (cheap, effective at small scale).
- **(b) RL:** **GRPO** (TRL) with verifiable rewards on math (answer-checking) — the
  modern, memory-light RLHF path. *Honest note:* RL gains are modest at 300M; we keep it
  as an instructive module and measure the delta.

### Stage 7 — Tool use & validation
- Define a **function-calling schema** (JSON), fold tool-call examples into SFT so the
  model emits/consumes `<tool_call>`. Validation: schema-conformance tests + a tool-call
  benchmark (BFCL-style subset).

### Stage 8 — Evaluation (every stage)
- **`lm-evaluation-harness`**: HellaSwag, ARC, MMLU, GSM8K, TruthfulQA, plus IFEval for
  instruct and a tool-call eval. Multimodal: VQAv2/COCO subset.
- Track scores per stage in a results table to *see* what each stage buys you.
- *Best practice:* fixed eval set, decontaminated, versioned; report tokens-seen vs score.

### Stage 9 — Train & host on cloud (GCP)
- **Terraform** for: GCS buckets (data/ckpts), Spot A100 VM with startup script that pulls
  repo + launches `torchrun`/accelerate, auto-checkpoint + auto-resume on preemption.
- **Serving:** **vLLM** (OpenAI-compatible API) on a small GPU VM; optional Cloud Run for
  the quantized version. Cost controls: Spot, auto-shutdown on idle, budget alerts.

### Stage 10 — Quantization for small-VRAM local use
- **GGUF + llama.cpp** (runs on CPU/Apple/low-VRAM) — primary path for "runs anywhere".
- **AWQ / GPTQ** (4-bit) for GPU serving via vLLM; **bitsandbytes** 4/8-bit for quick tests.
- Report quality-vs-size-vs-speed trade-offs (perplexity + a small eval at each bit-width).

### Stage 11 — Applications (showcasing real LLM use)
- **RAG:** chunk → embed (`sentence-transformers` or our model's embeddings) → FAISS/Chroma
  → retrieve → grounded answer with citations.
- **Tool use:** the function-calling model + a small tool registry (calculator, web, code).
- **Agentic:** a minimal ReAct/plan-act loop with memory using our served model.

### Stage 12 — "End-user readiness" factors
- Safety/refusals (light alignment + a refusal eval), system prompts, prompt-injection
  notes for RAG/agents, license/usage documentation, model card, latency/cost budgeting,
  reproducibility (seeds, data hashes), and guardrails for tool execution (sandboxing).

---

## 3. Tech stack (framework-first)
PyTorch · HF `datasets`/`tokenizers`/`transformers` · `accelerate` + DeepSpeed ·
FlashAttention-2 · TRL (SFT/DPO/GRPO) · lm-evaluation-harness · vLLM · llama.cpp/GGUF ·
AWQ/GPTQ/bitsandbytes · FAISS/Chroma · Terraform (GCP) · WandB.

## 4. Suggested build order (each chunk independently runnable)
1. Repo scaffold + configs + tests + data module
2. Tokenizer → 3. Base pretraining + eval
4. Quantization + local run (early, so you can *use* the base model fast)
5. SFT (instruct) + eval → 6. Reasoning + eval → 7. Tool use + eval
8. Vision toggle (from-scratch first, then SigLIP comparison)
9. GCP IaC + vLLM serving
10. Apps: RAG → tool agent → agentic
11. Model card + readiness checklist

## 5. Key risks / mitigations
- **Spot preemption** → checkpoint every N steps to GCS + auto-resume.
- **From-scratch vision underperforms** → toggle to SigLIP; compare honestly.
- **Budget overrun** → start single-GPU 124M, scale up only after pipeline works.
- **Eval contamination** → decontaminate; fixed versioned eval sets.

## 6. What I'd build first (on approval)
Scaffold the repo (structure above), the **data module**, **tokenizer trainer**, and the
**model definition + a 1-GPU smoke-train** that overfits a tiny batch — proving the whole
loop end-to-end before spending on multi-GPU. Then we scale each stage.
