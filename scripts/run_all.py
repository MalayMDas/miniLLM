"""Run the whole pipeline sequentially. One command, all stages, unattended.

    python scripts/run_all.py                 # local ~2h run (FineWeb-Edu stream, 6 GB GPU)
    python scripts/run_all.py --offline       # local, download a .bin once then no network
    python scripts/run_all.py --minipile      # ~1B model on MiniPile (~1.5B tok) — needs A100
    python scripts/run_all.py --smoke         # tiny + offline, ~1 min (verifies wiring)

RESTART / train more (unattended): just run it again — `pretrain.py` auto-resumes from
the latest checkpoint (model + optimizer + data position), trains further, then every
downstream stage re-runs on the new checkpoint.
    python scripts/run_all.py --minipile --add-steps 2000     # 2000 more steps

Stages: [prepare data] -> tokenizer -> pretrain -> eval -> SFT -> quantize -> sample.
Each runs as a subprocess so output streams live and import-order is clean.

REVIEW PROGRESS: in another terminal run `python scripts/status.py --watch`
or `tensorboard --logdir runs` (http://localhost:6006).
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


def profile(args) -> dict:
    """Resolve the config set for the chosen run. `prep` (or None) describes a one-time
    data download into a local .bin; `time_boxed` controls whether --pretrain-minutes
    caps training (the desktop run) vs running the config's step count (cloud)."""
    if args.smoke:
        return dict(tok_cfg="configs/model_tiny.yaml", pre_cfg="configs/pretrain_tiny.yaml",
                    sft_cfg="configs/sft_tiny.yaml", tok_path="artifacts/tok.json",
                    ckpt_dir="artifacts/ckpt_pretrain", sft_ckpt_dir="artifacts/ckpt_sft",
                    prep=None, time_boxed=False)
    if args.minipile:
        return dict(tok_cfg="configs/tokenizer_minipile.yaml",
                    pre_cfg="configs/pretrain_minipile.yaml",
                    sft_cfg="configs/sft_minipile.yaml",
                    tok_path="artifacts/tok_minipile.json",
                    ckpt_dir="artifacts/ckpt_pretrain_minipile",
                    sft_ckpt_dir="artifacts/ckpt_sft_minipile",
                    prep=dict(dataset="JeanKaddour/minipile", name="none",
                              out="data/minipile.bin",
                              tokens=args.prep_tokens or 1_500_000_000),
                    time_boxed=False)
    # local desktop run
    return dict(tok_cfg="configs/tokenizer_local.yaml",
                pre_cfg="configs/pretrain_local_offline.yaml" if args.offline
                        else "configs/pretrain_local.yaml",
                sft_cfg="configs/sft_local.yaml", tok_path="artifacts/tok_local.json",
                ckpt_dir="artifacts/ckpt_pretrain_local",
                sft_ckpt_dir="artifacts/ckpt_sft_local",
                prep=(dict(dataset="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                           out="data/fineweb_local.bin", tokens=args.prep_tokens or 100_000_000)
                      if args.offline else None),
                time_boxed=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny offline run to verify wiring")
    ap.add_argument("--offline", action="store_true",
                    help="local run from a pre-downloaded .bin (no network during training)")
    ap.add_argument("--minipile", action="store_true",
                    help="~1B model on MiniPile (~1.5B tokens); needs a 40-80 GB GPU")
    ap.add_argument("--pretrain-minutes", type=float, default=100.0)
    ap.add_argument("--add-steps", type=int, default=None,
                    help="resume the existing model and train this many MORE steps")
    ap.add_argument("--prep-tokens", type=int, default=None,
                    help="tokens to pre-download (overrides the profile default)")
    args = ap.parse_args()

    p = profile(args)
    total0 = time.perf_counter()
    print("TIP: watch progress with `python scripts/status.py --watch` "
          "or `tensorboard --logdir runs`.")

    # 1. tokenizer (skip if already trained)
    if Path(ROOT / p["tok_path"]).exists():
        print(f"[1] tokenizer exists ({p['tok_path']}) - skipping")
    else:
        run("1 tokenizer", [PY, "scripts/train_tokenizer.py", "--config", p["tok_cfg"]])

    # 1b. one-time data prep into a local .bin (offline / minipile)
    if p["prep"]:
        prep, out = p["prep"], ROOT / p["prep"]["out"]
        if out.exists():
            print(f"[1b] local corpus exists ({out.name}) - skipping prep")
        else:
            run("1b prepare-data", [PY, "scripts/prepare_data.py",
                                    "--tokenizer", p["tok_path"], "--dataset", prep["dataset"],
                                    "--name", prep["name"], "--out", prep["out"],
                                    "--tokens", str(prep["tokens"])])

    # 2. base pretraining (resumes from latest ckpt automatically if present)
    pre_cmd = [PY, "scripts/pretrain.py", "--config", p["pre_cfg"]]
    if args.add_steps is not None:
        pre_cmd += ["--add-steps", str(args.add_steps)]
    elif p["time_boxed"]:
        pre_cmd += ["--minutes", str(args.pretrain_minutes)]
    run("2 pretrain", pre_cmd)

    ckpt = find_latest(ROOT / p["ckpt_dir"])
    if ckpt is None:
        print("no checkpoint produced - aborting downstream stages")
        return
    ckpt = str(ckpt)
    print(f"latest checkpoint: {ckpt}")

    # 3. evaluate (fast local: perplexity + custom MCQ)
    run("3 evaluate", [PY, "scripts/evaluate.py", "--ckpt", ckpt, "--tokenizer", p["tok_path"]])

    # 4. instruct SFT (init from the base checkpoint)
    run("4 sft", [PY, "scripts/sft.py", "--config", p["sft_cfg"], "--init-from", ckpt])

    # 5. quantize (size/quality report)
    run("5 quantize", [PY, "scripts/quantize.py", "--ckpt", ckpt, "--tokenizer", p["tok_path"]])

    # 6. final sample from the SFT model
    sft_ckpt = find_latest(ROOT / p["sft_ckpt_dir"])
    if sft_ckpt:
        run("6 sample", [PY, "-c",
            "import sys; sys.path.insert(0,'src');"
            "import torch;"
            "from llmscratch.model import Decoder, ModelConfig;"
            "from llmscratch.tokenizer import build_tokenizer;"
            "from llmscratch.utils.checkpoint import load_checkpoint;"
            "from llmscratch.serve.generate import generate_chat;"
            f"tok=build_tokenizer({{'mode':'bpe','path':'{p['tok_path']}'}});"
            f"pl=torch.load(r'{sft_ckpt}',map_location='cpu',weights_only=False);"
            "m=Decoder(ModelConfig(**pl['model_config']));"
            f"load_checkpoint(r'{sft_ckpt}',m,map_location='cpu');m.eval();"
            "print(generate_chat(m,tok,[{'role':'user','content':'What is the capital of France?'}],max_new_tokens=40))"])

    print(f"\nALL STAGES DONE in {(time.perf_counter()-total0)/60:.1f} min. "
          f"Review: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
