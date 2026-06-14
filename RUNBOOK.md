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

## 11. Quantize + serve
```bash
python scripts/quantize.py --ckpt /artifacts/ckpt_sft/step_XXXX.pt   # size/quality report
# serve our model (understanding) ...
python -m llmscratch.serve.api --ckpt <ckpt> --tokenizer artifacts/tok_32k.json
# ... or for throughput, GGUF + llama.cpp (laptop) / vLLM (GPU): see PLAN.md Stage 10
```

---

## 12. Cost controls (do these or you will overspend)
- **Always use Spot/preemptible** + `sky jobs launch` (auto-recovery).
- **Checkpoint to object storage** every `ckpt_every`; `find_latest` resume is built in.
- **Autostop idle clusters**: `sky autostop -i 10 llm-train` (or `sky down` when done).
- **Set a provider budget alert** (e.g. $400) — belt and suspenders.
- Start each stage at **tiny scale on 1 GPU** to confirm it runs before booking 8.

## 13. Troubleshooting
| Symptom | Cause → Fix |
|---|---|
| CUDA OOM | batch too big → lower `batch_size`, raise `grad_accum`; enable grad checkpointing; if model doesn't fit, switch DDP→**FSDP/DeepSpeed** via `accelerate config` (loop unchanged). |
| Job killed mid-run | Spot preemption → relaunch same command; it auto-resumes from latest ckpt. |
| `loss` is NaN | LR too high / no warmup → lower `lr`, increase `warmup_steps`; confirm bf16 (not fp16). |
| Segfault on dataset load (Windows) | torch imported before pyarrow → already guarded in our scripts (`import datasets` first). Non-issue on Linux. |
| `tokenizer mismatch` / garbage | eval/serve used a different tokenizer than training → always pass the *same* `tok_32k.json`. |
| Throughput low | small batch / no FlashAttention → raise batch, ensure bf16 + SDPA flash kernels; check `tokens_per_sec` in W&B. |
| Resume starts at step 0 | `ckpt_dir` not persisted (Spot wiped local disk) → point it at the mounted bucket. |

## 14. Teardown
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
