"""Generate a MODEL_CARD.md for a checkpoint (end-user readiness, Stage 12).

    python scripts/model_card.py --ckpt artifacts/ckpt_sft_local/step_X.pt \
        --name miniLLM-41M --out MODEL_CARD.md [--eval results.json]

Pulls the architecture from the checkpoint and fills a card with intended use,
training data, limitations, safety, and license. Optionally embeds an eval-results
JSON (e.g. from benchmark.py / lm_eval_run.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

TEMPLATE = """# {name}

A small Llama-style language model built from scratch with the `llmscratch` pipeline
(learning project). **Not a production model.**

## Architecture
- Params: **~{params:.0f}M**  ·  decoder-only transformer (RoPE, RMSNorm, SwiGLU, GQA)
- Layers: {n_layers}  ·  dim: {dim}  ·  heads: {n_heads} (KV heads: {n_kv_heads})  ·  context: {max_seq_len}
- Vocab: {vocab_size} (byte-level BPE)

## Training data
{data}

## Intended use
Educational / research demonstration of the full LLM lifecycle (pretrain, SFT,
reasoning, tool use, eval, quantize, serve). Suitable for experimentation, not for
factual, safety-critical, or production use.

## Evaluation
{eval}

## Limitations & risks
- **Small + undertrained** → frequently wrong, low factual reliability, limited reasoning.
- May produce **hallucinated, biased, or unsafe** content; outputs are not filtered.
- English-centric; weak on code/math unless trained with those mixes.
- Tool/agent use can execute actions — run tools sandboxed (see READINESS.md).

## Safety
Light refusal alignment via SFT on refusal examples; measured with
`llmscratch.eval.safety`. This is **not** a robust safety guarantee — add a moderation
layer for any real deployment.

## License & provenance
- Code: see repository LICENSE. Training data: per-dataset licenses (e.g. FineWeb-Edu
  ODC-BY, MiniPile, Wikipedia CC-BY-SA) — verify before redistribution.
- Reproducibility: run id = git SHA + config hash; checkpoint: `{ckpt}`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="miniLLM")
    ap.add_argument("--data", default="Pretrained on a FineWeb-Edu / MiniPile sample; "
                    "instruct + reasoning + tool SFT on small mixtures. See configs/.")
    ap.add_argument("--eval", default=None, help="path to an eval-results JSON to embed")
    ap.add_argument("--out", default="MODEL_CARD.md")
    args = ap.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    c = payload["model_config"]
    # param estimate from config (no need to build the model)
    hid = ((int(8 / 3 * c["dim"]) + 255) // 256) * 256
    hd = c["dim"] // c["n_heads"]
    per = 2 * c["dim"] ** 2 + 2 * c["dim"] * c["n_kv_heads"] * hd + 3 * c["dim"] * hid
    params = (c["n_layers"] * per + c["vocab_size"] * c["dim"]) / 1e6

    eval_txt = "_(none provided)_"
    if args.eval and Path(args.eval).exists():
        results = json.loads(Path(args.eval).read_text())
        eval_txt = "```json\n" + json.dumps(results, indent=2) + "\n```"

    card = TEMPLATE.format(name=args.name, params=params, n_layers=c["n_layers"],
                           dim=c["dim"], n_heads=c["n_heads"], n_kv_heads=c["n_kv_heads"],
                           max_seq_len=c["max_seq_len"], vocab_size=c["vocab_size"],
                           data=args.data, eval=eval_txt, ckpt=Path(args.ckpt).name)
    Path(args.out).write_text(card, encoding="utf-8")
    print(f"wrote {args.out} (~{params:.0f}M params)")


if __name__ == "__main__":
    main()
