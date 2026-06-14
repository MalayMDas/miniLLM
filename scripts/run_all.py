"""Run the whole pipeline sequentially — sized for a ~2h local run on a 6 GB GPU.

    python scripts/run_all.py                 # local 2h run (real FineWeb-Edu data)
    python scripts/run_all.py --pretrain-minutes 100
    python scripts/run_all.py --smoke          # tiny + offline, ~1 min (verifies wiring)

RESTART / train more (unattended): just run it again. `pretrain.py` auto-resumes from
the latest checkpoint (model + optimizer state), trains further, then every downstream
stage re-runs on the new checkpoint. Two ways to say "train more":
    python scripts/run_all.py --pretrain-minutes 30    # 30 more wall-clock minutes
    python scripts/run_all.py --add-steps 2000         # 2000 more steps (fresh warmup+cosine)

Stages, in order:  tokenizer -> pretrain (time-boxed) -> eval -> SFT -> quantize ->
sample. Each runs as a subprocess so output streams live and import-order is clean.

REVIEW PROGRESS while it runs: open a second terminal and run
    tensorboard --logdir runs       (then open http://localhost:6006)
You'll see train/loss, perplexity, lr, grad-norm, tokens/sec, and sample text update
live for the `pretrain_local` (and `sft_local`) runs.

The `--smoke` path uses the tiny offline configs so you can confirm every stage wires
together in ~a minute before committing two hours.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llmscratch.utils import find_latest  # noqa: E402

PY = sys.executable


def run(title: str, cmd: list) -> None:
    print("\n" + "=" * 70 + f"\n  {title}\n  $ {' '.join(str(c) for c in cmd)}\n" + "=" * 70)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"  [{title}] done in {(time.perf_counter()-t0)/60:.1f} min")


# (tokenizer_cfg, pretrain_cfg, sft_cfg, tokenizer_path, pretrain_ckpt_dir)
LOCAL = ("configs/tokenizer_local.yaml", "configs/pretrain_local.yaml",
         "configs/sft_local.yaml", "artifacts/tok_local.json",
         "artifacts/ckpt_pretrain_local")
SMOKE = ("configs/model_tiny.yaml", "configs/pretrain_tiny.yaml",
         "configs/sft_tiny.yaml", "artifacts/tok.json",
         "artifacts/ckpt_pretrain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny offline run to verify wiring")
    ap.add_argument("--pretrain-minutes", type=float, default=100.0)
    ap.add_argument("--add-steps", type=int, default=None,
                    help="resume the existing model and train this many MORE steps")
    args = ap.parse_args()

    tok_cfg, pre_cfg, sft_cfg, tok_path, ckpt_dir = SMOKE if args.smoke else LOCAL
    total0 = time.perf_counter()
    print("TIP: in another terminal run `tensorboard --logdir runs` to watch progress.")

    # 1. tokenizer (skip if already trained)
    if Path(ROOT / tok_path).exists():
        print(f"[1/6] tokenizer exists ({tok_path}) - skipping")
    else:
        run("1/6 tokenizer", [PY, "scripts/train_tokenizer.py", "--config", tok_cfg])

    # 2. base pretraining (resumes from latest ckpt automatically if present)
    pre_cmd = [PY, "scripts/pretrain.py", "--config", pre_cfg]
    if args.add_steps is not None:
        pre_cmd += ["--add-steps", str(args.add_steps)]   # explicit "train N more"
    elif not args.smoke:
        pre_cmd += ["--minutes", str(args.pretrain_minutes)]
    run("2/6 pretrain", pre_cmd)

    ckpt = find_latest(ROOT / ckpt_dir)
    if ckpt is None:
        print("no checkpoint produced — aborting downstream stages")
        return
    ckpt = str(ckpt)
    print(f"latest checkpoint: {ckpt}")

    # 3. evaluate (fast local: perplexity + custom MCQ)
    run("3/6 evaluate", [PY, "scripts/evaluate.py", "--ckpt", ckpt, "--tokenizer", tok_path])

    # 4. instruct SFT (init from the base checkpoint)
    run("4/6 sft", [PY, "scripts/sft.py", "--config", sft_cfg, "--init-from", ckpt])

    # 5. quantize (size/quality report)
    run("5/6 quantize", [PY, "scripts/quantize.py", "--ckpt", ckpt, "--tokenizer", tok_path])

    # 6. final sample from the SFT model
    sft_ckpt = find_latest(ROOT / ("artifacts/ckpt_sft_local" if not args.smoke
                                   else "artifacts/ckpt_sft"))
    if sft_ckpt:
        run("6/6 sample", [PY, "-c",
            "import sys; sys.path.insert(0,'src');"
            "import torch;"
            "from llmscratch.model import Decoder, ModelConfig;"
            "from llmscratch.tokenizer import build_tokenizer;"
            "from llmscratch.utils.checkpoint import load_checkpoint;"
            "from llmscratch.serve.generate import generate_chat;"
            f"tok=build_tokenizer({{'mode':'bpe','path':'{tok_path}'}});"
            f"p=torch.load(r'{sft_ckpt}',map_location='cpu',weights_only=False);"
            "m=Decoder(ModelConfig(**p['model_config']));"
            f"load_checkpoint(r'{sft_ckpt}',m,map_location='cpu');m.eval();"
            "print(generate_chat(m,tok,[{'role':'user','content':'What is the capital of France?'}],max_new_tokens=40))"])

    print(f"\nALL STAGES DONE in {(time.perf_counter()-total0)/60:.1f} min. "
          f"Review curves: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
