# Cloud Training Runbook

End-to-end steps to train the real model on a cheap multi-GPU provider, stage by
stage, with cost controls and recovery. Everything references scripts/configs that
already exist in this repo. Target: **< $500**, cheaper provider (Lambda / RunPod /
vast.ai), Spot + checkpointing.

> Golden rule: **prove each stage on the tiny config locally first** (`make smoke`,
> `python scripts/demo.py`), then run the real config in the cloud. The only
> differences are model size, data source, and GPU count — all in YAML.

---

## 0. Budget & timeline (rough, 8×A100-40GB ≈ $10–14/hr Spot)

| Stage | Script / config | Wall-clock | ~Cost |
|---|---|---|---|
| Tokenizer (32k) | `train_tokenizer.py` + `tokenizer_32k.yaml` | ~20 min (1 GPU/CPU) | ~$2 |
| Base pretrain (~300M, 10–20B tok) | `pretrain.py` + `pretrain_300m.yaml` | ~8–14 h | ~$120–200 |
| Eval (base) | `lm_eval_run.py` | ~20 min | ~$3 |
| Instruct SFT | `sft.py` + `sft_*.yaml` | ~1–2 h | ~$15–30 |
| Reasoning (CoT SFT + GRPO) | `sft.py` + `align.grpo` | ~2–4 h | ~$30–60 |
| Tool SFT | `sft.py` + `sft_tools_*.yaml` | ~1 h | ~$12 |
| Multimodal (optional) | `train_vision.py` | ~2–4 h | ~$30–60 |
| Eval per stage + quantize | `benchmark.py` / `quantize.py` | ~1 h total | ~$10 |
| **Total** | | **~1–1.5 days** | **~$220–380** |

Leaves headroom under $500 for retries. Drop to **1×A100 + 124M** to cut the base
run to ~$40 if budget is tight (same configs, smaller numbers).

---

## 1. Pre-flight checklist

- [ ] Provider account with GPU quota (Lambda / RunPod / vast.ai).
- [ ] `HF_TOKEN` (HuggingFace) — FineWeb-Edu is public, but set it to avoid rate limits.
- [ ] `WANDB_API_KEY` — hosted tracking for the long run.
- [ ] A persistent **object-store bucket** (GCS/S3) for checkpoints — *mandatory* for Spot.
- [ ] Local: `python scripts/demo.py` passes and `pytest` is green.
- [ ] Decide GPU count `N` and set `grad_accum` so effective batch is sane
      (effective batch = `batch_size × grad_accum × N`).

---

## 2. Provision + environment

### Option A — SkyPilot (recommended: cheapest spot, auto-recovery, 1 command)
```bash
pip install "skypilot[aws,gcp,runpod]"
sky check                                   # configure a cloud once
# edit infra/sky/train.yaml: set accelerators + the run: command for the stage
sky jobs launch -n llm-train infra/sky/train.yaml     # managed spot, auto-resume
sky jobs queue          # watch        sky jobs logs llm-train      # tail
sky down llm-train                          # STOP PAYING when done
```
Mount the checkpoint bucket so artifacts survive preemption — add to `train.yaml`:
```yaml
file_mounts:
  /artifacts:
    source: gs://YOUR-BUCKET/llm-artifacts   # or s3://
    mode: MOUNT
```
…and point every config's `ckpt_dir` under `/artifacts/...`.

### Option B — rented pod, no Docker (fastest manual start)
Launch a PyTorch+CUDA template pod, then:
```bash
git clone <your-repo> && cd llm-from-scratch
export HF_TOKEN=...  WANDB_API_KEY=...
bash infra/cloud_setup.sh          # pinned install + GPU sanity + smoke-train
wandb login $WANDB_API_KEY
```

### Option C — Docker (most reproducible / for serving)
```bash
docker build -t llm:dev .
docker run --gpus all -e HF_TOKEN -e WANDB_API_KEY -v $PWD/artifacts:/workspace/artifacts llm:dev <cmd>
```

---

## 2.5 Data persistence & caching (the iterate-many-times compute saver)

**Goal:** download/tokenize the corpus **once, ever** — then every run, pod, rank, and
spot-resume reads it locally with no network. This is the single biggest way to stop
paying GPU-hours for downloads.

**Do NOT bake data into the Docker image** (huge images, rebuilt on every code change,
re-pulled per node). Keep code in the image, **data on persistent storage, mounted at
runtime.**

Where "persistent storage" lives on cheap providers:
- **RunPod** → **Network Volume** (persists across pods) — mount at `/data-cache`.
- **Lambda Cloud** → attachable **persistent filesystem** (region-dependent).
- **vast.ai** → rentable **persistent volume** on the host.
- **Universal / provider-independent** → your own cheap object store: **Cloudflare R2**
  (zero egress) or **Backblaze B2**; `rclone copy` to the node's local disk at startup,
  or mount via SkyPilot `file_mounts` on AWS/GCP/Azure.

Two things to put on the persistent mount:
1. **HF cache** — `export HF_HOME=/data-cache/hf` so `datasets`/model downloads are reused.
2. **Pre-tokenized `.bin`** — `prepare_data.py` once → train offline forever after:
```bash
# one-time (writes to the persistent mount):
python scripts/prepare_data.py --tokenizer /data-cache/tok_32k.json \
    --tokens 3000000000 --out /data-cache/fineweb.bin      # ~6 GB for 3B tokens (uint16)
# then in your pretrain config:  data.source: bin,  data.bin_path: /data-cache/fineweb.bin
```
`infra/sky/train.yaml` already shows the bucket mounts + `HF_HOME` + a guarded one-time
prep (re-runs skip it). Also `pip install hf_xet` for faster transfer.

**Streaming vs offline tradeoff:** streaming (`source: hf`) needs no prep but re-downloads
each run and can stall on the network; set `data.num_workers: 4` so background workers
prefetch+overlap with compute (Linux). Offline (`source: bin`) prepays once and trains
with zero network. **For iterating many times, prefer the `.bin` on a persistent mount.**

**Budget option — small corpus, fixed model size:** to fit a hard budget without shrinking
the model, train on a smaller corpus (fewer tokens ≈ proportionally less cost). Example
wired in: **MiniPile** (~1.5B tokens) for a ~1B model — `configs/tokenizer_minipile.yaml`
+ `configs/pretrain_minipile.yaml` (~$25–35 on 1×A100-80GB for ~1 epoch). MiniPile has no
config name, so `--name none` (or omit `hf_name`) — handled by the scripts. Honest: 1.5B
tokens ≪ Chinchilla's ~20B for 1B params, so the model is **undertrained** (weak benchmarks);
a smaller model trained to ~20×params tokens would be better per dollar, but this keeps 1B.

## 3. Stage 1 — Tokenizer (32k, on a FineWeb-Edu sample)
```bash
python scripts/train_tokenizer.py --config configs/tokenizer_32k.yaml
# -> artifacts/tok_32k.json   (point pretrain_300m.yaml tokenizer.path at it)
```
Sanity: `python -m llmscratch.tokenizer.compare --bpe artifacts/tok_32k.json`.

## 4. Stage 2 — Base pretraining (multi-GPU)
```bash
# set tokenizer.path: artifacts/tok_32k.json and ckpt_dir under the mounted bucket
torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain_300m.yaml
```
- Logs to W&B (`logging.backend: wandb`); watch `train/loss`, `tokens_per_sec`.
- Checkpoints every `ckpt_every` steps to `ckpt_dir`. **On preemption, just relaunch
  the same command** — `pretrain.py` calls `find_latest` and resumes automatically.
- Tune `train.steps` to your token budget (Chinchilla ≈ 20× params in tokens → ~6B
  for 300M as a floor; more data is better for small models).

## 5. Stage 3 — Evaluate the base model (official numbers)
```bash
pip install lm-eval
python scripts/lm_eval_run.py --ckpt /artifacts/ckpt_pretrain_300m/step_XXXX.pt \
  --tokenizer artifacts/tok_32k.json \
  --tasks hellaswag,arc_easy,openbookqa,piqa,winogrande,sciq,lambada_openai --limit 1000
```
Record the numbers (see the tracking table in §10). Expect HellaSwag/ARC-easy/PIQA
above chance; MMLU/GSM8K near chance until later stages + scale.

## 5.5 Fetch real post-training data (one-time, optional but recommended)
The pipeline uses tiny placeholders unless you prepare real data; `run_all` then
auto-uses these files:
```bash
python scripts/prepare_instruct.py   # UltraChat-200k -> data/instruct.jsonl
python scripts/prepare_tools.py      # xLAM-60k       -> data/tools.jsonl
python scripts/prepare_reason.py     # GSM8K CoT      -> data/reason.jsonl
```
The instruct SFT stage trains on a **merged mix** (instruct + tools + safety) in one
pass; reasoning is a separate CoT pass. Raise `data.max_len` in the sft configs for the
longer tool-schema sequences on real runs.

## 6. Stage 4 — Instruct SFT
Build a chat config from real data (UltraChat / OpenHermes / Tulu subset → jsonl in
the `{"messages":[...]}` format), set `init_from` to the base checkpoint:
```bash
python scripts/sft.py --config configs/sft_300m.yaml      # copy sft_tiny.yaml, scale dims
```
Then re-run §5 + IFEval: add `ifeval` to `--tasks`.

## 7. Stage 5 — Reasoning
1. **CoT distillation** (cheap, do first): SFT on reasoning traces (OpenMathReasoning
   / OpenR1) formatted with `align.reasoning.format_cot_turn` → run `sft.py`.
2. **GRPO** (verifiable rewards on math): drive `align.grpo.grpo_step` with a
   reward that checks the final answer (reuse `eval/tasks/gsm8k.extract_pred`).
Re-eval GSM8K; track the delta from CoT→GRPO.

## 8. Stage 6 — Tool use SFT
```bash
python scripts/sft.py --config configs/sft_tools_300m.yaml   # scale sft_tools_tiny.yaml
```
Eval with `scripts/benchmark.py --tasks bfcl` against a real BFCL slice.

## 9. Stage 7 — Multimodal (optional)
```bash
python scripts/train_vision.py --phase 1     # projector alignment (LLaVA phase 1)
python scripts/train_vision.py --phase 2     # instruction tune
```
For real capability flip `vision/encoder.py` to the pretrained SigLIP toggle.
Eval with `llmscratch.eval.tasks.evaluate_vqa` on a VQAv2 slice (needs COCO images).

## 10. Track every stage (the artifact that tells the story)

| Stage | HellaSwag | ARC-e | PIQA | GSM8K | IFEval | BFCL |
|---|---|---|---|---|---|---|
| base | | | | | — | — |
| +instruct | | | | | | |
| +reasoning | | | | | | |
| +tools | | | | | | |

Fill from `lm_eval_run.py` + `benchmark.py`. This per-stage table is the single most
useful output — it shows what each stage *buys*.

## 10.5 Mixing in code data (optional)
A small fraction of code helps the model read/write code. Mix datasets into one `.bin`:
```bash
python scripts/prepare_data.py --tokenizer artifacts/tok_32k.json \
  --datasets "HuggingFaceFW/fineweb-edu,bigcode/the-stack-smol" \
  --names "sample-10BT,none" --text-fields "text,content" --weights "0.9,0.1" \
  --tokens 3000000000 --out /data-cache/mix.bin
```
(`weighted_interleave` governs the ratio over the consumed prefix; code uses the
`content` field, not `text`.)

## 11. Export for real serving / low-VRAM (HF → vLLM / GGUF)
Our checkpoint exports to a **genuine HF `LlamaForCausalLM`** (verified numerically
identical), which unlocks the standard toolchain:
```bash
# HF folder -> serve with vLLM, or eval with lm-eval-harness:
python scripts/export_hf.py --ckpt <ckpt> --tokenizer artifacts/tok_32k.json --out artifacts/hf_model
vllm serve artifacts/hf_model                                  # OpenAI API, high throughput
lm_eval --model hf --model_args pretrained=artifacts/hf_model --tasks hellaswag

# GGUF for llama.cpp / Ollama / LM Studio (CPU / Apple / low-VRAM):
git clone https://github.com/ggerganov/llama.cpp
python scripts/export_gguf.py --ckpt <ckpt> --tokenizer artifacts/tok_32k.json \
  --llama-cpp ./llama.cpp --out artifacts/model.gguf --quantize Q4_K_M
```

## 12. Quantize + serve (our transparent path)
```bash
python scripts/quantize.py --ckpt /artifacts/ckpt_sft/step_XXXX.pt   # size/quality report
# serve our model (understanding) ...
python -m llmscratch.serve.api --ckpt <ckpt> --tokenizer artifacts/tok_32k.json
# ... or for throughput, GGUF + llama.cpp (laptop) / vLLM (GPU): see PLAN.md Stage 10
```

---

## 13. Cost controls (do these or you will overspend)
- **Always use Spot/preemptible** + `sky jobs launch` (auto-recovery).
- **Checkpoint to object storage** every `ckpt_every`; `find_latest` resume is built in.
- **Autostop idle clusters**: `sky autostop -i 10 llm-train` (or `sky down` when done).
- **Set a provider budget alert** (e.g. $400) — belt and suspenders.
- Start each stage at **tiny scale on 1 GPU** to confirm it runs before booking 8.

## 14. Troubleshooting
| Symptom | Cause → Fix |
|---|---|
| CUDA OOM | batch too big → lower `batch_size`, raise `grad_accum`; enable grad checkpointing; if model doesn't fit, switch DDP→**FSDP/DeepSpeed** via `accelerate config` (loop unchanged). |
| Run "looks stuck" (no step logs) | model too big for the GPU → on Windows the driver spills to slow shared system RAM instead of OOM-ing. `pretrain.py` now runs a **VRAM preflight** that aborts with a clear message; use a smaller profile (`--minipile-local`) or `--allow-oversize` to override. Also note: `source: bin` does NOT download (banner shows OFFLINE). |
| Job killed mid-run | Spot preemption → relaunch same command; it auto-resumes from latest ckpt. |
| `loss` is NaN | LR too high / no warmup → lower `lr`, increase `warmup_steps`; confirm bf16 (not fp16). |
| Segfault on dataset load (Windows) | torch imported before pyarrow → already guarded in our scripts (`import datasets` first). Non-issue on Linux. |
| `tokenizer mismatch` / garbage | eval/serve used a different tokenizer than training → always pass the *same* `tok_32k.json`. |
| Throughput low | small batch / no FlashAttention → raise batch, ensure bf16 + SDPA flash kernels; check `tokens_per_sec` in W&B. |
| Resume starts at step 0 | `ckpt_dir` not persisted (Spot wiped local disk) → point it at the mounted bucket. |

## 15. Teardown
```bash
sky down llm-train          # or terminate the pod in the provider console
```
Confirm the GPU is released (you pay until it is). Checkpoints persist in the bucket.

---

### One-screen quick path
```bash
# on the GPU box, repo cloned, keys exported:
bash infra/cloud_setup.sh
python scripts/train_tokenizer.py --config configs/tokenizer_32k.yaml
torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain_300m.yaml
python scripts/lm_eval_run.py --ckpt /artifacts/ckpt_pretrain_300m/step_XXXX.pt --tokenizer artifacts/tok_32k.json
python scripts/sft.py --config configs/sft_300m.yaml
# ... reasoning, tools, quantize, serve ...
sky down llm-train
```
