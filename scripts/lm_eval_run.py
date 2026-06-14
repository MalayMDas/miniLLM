"""Run EleutherAI lm-evaluation-harness on our checkpoint (official numbers).

    pip install lm-eval
    python scripts/lm_eval_run.py --ckpt artifacts/ckpt_pretrain_300m/step_0050000.pt \
        --tokenizer artifacts/tok_32k.json \
        --tasks hellaswag,arc_easy,openbookqa,piqa,winogrande,sciq,lambada_openai,gsm8k \
        --limit 200

All local: the harness downloads task data once from the HF Hub, scoring runs on
your GPU. BFCL + VQAv2 are not in lm-eval; use scripts/benchmark.py for those.

Small-model note: pick tasks with signal at your scale (HellaSwag/ARC-easy/PIQA/
WinoGrande/SciQ/LAMBADA). MMLU/GSM8K stay near chance until you scale + add reasoning.
"""
from __future__ import annotations

# pyarrow/datasets before torch (Windows DLL clash); harmless on Linux.
try:
    import datasets  # noqa: F401
except Exception:
    pass

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from llmscratch.eval.lm_eval_adapter import build_lm  # noqa: E402


SMALL_MODEL_SUITE = "hellaswag,arc_easy,openbookqa,piqa,winogrande,sciq,lambada_openai"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tok_32k.json")
    ap.add_argument("--tasks", default=SMALL_MODEL_SUITE)
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="examples/task (None=full)")
    ap.add_argument("--output", default=None, help="write results json here")
    args = ap.parse_args()

    from lm_eval import simple_evaluate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = build_lm(args.ckpt, args.tokenizer, device=device)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    results = simple_evaluate(model=lm, tasks=tasks, num_fewshot=args.num_fewshot,
                              limit=args.limit)

    print(f"\n=== lm-eval: {Path(args.ckpt).name} ===")
    for task, metrics in results["results"].items():
        line = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                         if isinstance(v, float))
        print(f"  {task:18}: {line}")

    if args.output:
        Path(args.output).write_text(json.dumps(results["results"], indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
