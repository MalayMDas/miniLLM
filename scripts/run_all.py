"""Run the whole pipeline sequentially. One command, all stages, unattended.

    python scripts/run_all.py                 # local ~2h run (FineWeb-Edu stream, 6 GB GPU)
    python scripts/run_all.py --offline       # local, download a .bin once then no network
    python scripts/run_all.py --minipile-local # ~41M model on MiniPile, fits a 6 GB GPU
    python scripts/run_all.py --minipile      # ~1B model on MiniPile (~1.5B tok) — needs A100
    python scripts/run_all.py --smoke         # tiny + offline, ~1 min (verifies wiring)

RESTART / train more (unattended): just run it again — `pretrain.py` auto-resumes from
the latest checkpoint (model + optimizer + data position), trains further, then every
downstream stage re-runs on the new checkpoint.
    python scripts/run_all.py --minipile --add-steps 2000     # 2000 more steps

Stages: [prepare data] -> tokenizer -> pretrain -> eval -> instruct SFT (merged
instruct + tool-use + safety) -> reasoning (CoT distillation) -> quantize -> sample.
Each runs as a subprocess so output streams live and import-order is clean. Instruct,
tool-use, and reasoning use REAL data when prepared (prepare_instruct.py /
prepare_tools.py / prepare_reason.py), else tiny offline placeholders. The reasoning
pass trains on CoT (<think>...</think>), so the final model emits reasoning tags.

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
from llmscratch.utils import find_latest, load_config  # noqa: E402

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
        return dict(name="smoke", tok_cfg="configs/model_tiny.yaml",
                    pre_cfg="configs/pretrain_tiny.yaml",
                    sft_cfg="configs/sft_tiny.yaml", tok_path="artifacts/tok.json",
                    ckpt_dir="artifacts/ckpt_pretrain", sft_ckpt_dir="artifacts/ckpt_sft",
                    reason_ckpt_dir="artifacts/ckpt_reason", prep=None,
                    real_posttrain=False, time_boxed=False)
    if args.minipile or args.minipile_local:
        local = args.minipile_local
        sfx = "_local" if local else ""
        # local: pretrain on a MiniPile + code MIX and use REAL post-training data.
        prep = (dict(mix=True,
                     datasets="JeanKaddour/minipile,bigcode/the-stack-smol",
                     names="none,none", data_dirs=",data/python",
                     text_fields="text,content", weights="0.9,0.1",
                     out="data/minipile_code.bin", tokens=args.prep_tokens or 1_500_000_000)
                if local else
                dict(dataset="JeanKaddour/minipile", name="none",
                     out="data/minipile.bin", tokens=args.prep_tokens or 1_500_000_000))
        return dict(name="minipile-local" if local else "minipile",
                    tok_cfg="configs/tokenizer_minipile.yaml",
                    pre_cfg=f"configs/pretrain_minipile{'_local' if local else ''}.yaml",
                    sft_cfg=f"configs/sft_minipile{'_local' if local else ''}.yaml",
                    tok_path="artifacts/tok_minipile.json",
                    ckpt_dir=f"artifacts/ckpt_pretrain_minipile{sfx}",
                    sft_ckpt_dir=f"artifacts/ckpt_sft_minipile{sfx}",
                    reason_ckpt_dir=f"artifacts/ckpt_reason_minipile{sfx}",
                    prep=prep,
                    real_posttrain=local,   # fetch real instruct/tools/reasoning data
                    time_boxed=local)       # local is time-boxed; cloud runs the step count
    # local desktop run
    return dict(name="local-offline" if args.offline else "local-streaming",
                tok_cfg="configs/tokenizer_local.yaml",
                pre_cfg="configs/pretrain_local_offline.yaml" if args.offline
                        else "configs/pretrain_local.yaml",
                sft_cfg="configs/sft_local.yaml", tok_path="artifacts/tok_local.json",
                ckpt_dir="artifacts/ckpt_pretrain_local",
                sft_ckpt_dir="artifacts/ckpt_sft_local",
                reason_ckpt_dir="artifacts/ckpt_reason_local",
                prep=(dict(dataset="HuggingFaceFW/fineweb-edu", name="sample-10BT",
                           out="data/fineweb_local.bin", tokens=args.prep_tokens or 100_000_000)
                      if args.offline else None),
                real_posttrain=False, time_boxed=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny offline run to verify wiring")
    ap.add_argument("--offline", action="store_true",
                    help="local run from a pre-downloaded .bin (no network during training)")
    ap.add_argument("--minipile", action="store_true",
                    help="~1B model on MiniPile (~1.5B tokens); needs a 40-80 GB GPU")
    ap.add_argument("--minipile-local", action="store_true",
                    help="~41M model on MiniPile; fits a 6 GB GPU, reuses data/minipile.bin")
    ap.add_argument("--pretrain-minutes", type=float, default=100.0)
    ap.add_argument("--add-steps", type=int, default=None,
                    help="resume the existing model and train this many MORE steps")
    ap.add_argument("--prep-tokens", type=int, default=None,
                    help="tokens to pre-download (overrides the profile default)")
    ap.add_argument("--instruct-data", default=None,
                    help="instruct chat jsonl (default: data/instruct.jsonl if present "
                         "from prepare_instruct.py, else the tiny placeholder)")
    ap.add_argument("--tools-data", default=None,
                    help="tool-call jsonl (default: data/tools.jsonl if present from "
                         "prepare_tools.py, else the tiny placeholder)")
    ap.add_argument("--reason-data", default=None,
                    help="CoT jsonl for the reasoning stage. Default: data/reason.jsonl if "
                         "present (run scripts/prepare_reason.py to fetch real GSM8K CoT), "
                         "else the tiny data/sample_reason.jsonl placeholder.")
    args = ap.parse_args()

    p = profile(args)
    total0 = time.perf_counter()

    # Make the active dataset explicit (so it's obvious which corpus / source is used).
    d = load_config(ROOT / p["pre_cfg"])["data"]
    if d["source"] == "hf":
        where = f"STREAMING {d['hf_dataset']} ({d.get('hf_name')})"
        hint = ("   -> network needed; expect occasional CDN retry warnings. Use --offline "
                "(download once) or --minipile to avoid them; `pip install hf_xet` speeds it up.")
    elif d["source"] == "bin":
        where = f"OFFLINE {d['bin_path']} (no network)"
        hint = ""
    else:
        where = f"LOCAL {d.get('corpus')}"
        hint = ""
    print(f"\n[profile: {p['name']}]  pretrain={p['pre_cfg']}  data: {where}")
    if hint:
        print(hint)
    print("TIP: watch progress with `python scripts/status.py --watch` "
          "or `tensorboard --logdir runs`.")

    # 1. tokenizer (skip if already trained)
    if Path(ROOT / p["tok_path"]).exists():
        print(f"[1] tokenizer exists ({p['tok_path']}) - skipping")
    else:
        run("1 tokenizer", [PY, "scripts/train_tokenizer.py", "--config", p["tok_cfg"]])

    # 1b. one-time pretraining-data prep into a local .bin (offline / minipile [+ code])
    if p["prep"]:
        prep, out = p["prep"], ROOT / p["prep"]["out"]
        if out.exists():
            print(f"[1b] pretrain corpus exists ({out.name}) - skipping prep")
        else:
            cmd = [PY, "scripts/prepare_data.py", "--tokenizer", p["tok_path"],
                   "--out", prep["out"], "--tokens", str(prep["tokens"])]
            if prep.get("mix"):     # MiniPile + code
                cmd += ["--datasets", prep["datasets"], "--names", prep["names"],
                        "--weights", prep["weights"], "--text-fields", prep["text_fields"],
                        "--data-dirs", prep["data_dirs"]]
            else:
                cmd += ["--dataset", prep["dataset"], "--name", prep["name"]]
            run("1b prepare-data", cmd)

    # 1c. fetch REAL post-training data (instruct/tools/reasoning) once, if the profile
    # wants it. run_all auto-uses these files in the SFT mix + reasoning stage below.
    if p.get("real_posttrain"):
        for script, outf in [("prepare_instruct.py", "data/instruct.jsonl"),
                             ("prepare_tools.py", "data/tools.jsonl"),
                             ("prepare_reason.py", "data/reason.jsonl")]:
            if (ROOT / outf).exists():
                print(f"[1c] {outf} exists - skipping {script}")
            else:
                run(f"1c {script}", [PY, f"scripts/{script}"])

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

    # 4. instruct SFT on a MERGED mix: instruct + tool-use + safety (one pass, so no
    # catastrophic forgetting). Each component uses real prepared data if present
    # (prepare_instruct.py / prepare_tools.py), else the tiny offline placeholder.
    def _pick(real, placeholder):
        return real if (ROOT / real).exists() else placeholder
    mix = [args.instruct_data or _pick("data/instruct.jsonl", "data/sample_chat.jsonl"),
           args.tools_data or _pick("data/tools.jsonl", "data/sample_tools_chat.jsonl"),
           "data/sample_safety.jsonl"]
    mix = [m for m in mix if (ROOT / m).exists()]
    print(f"   instruct SFT mix (instruct + tools + safety): {mix}")
    run("4 instruct-sft", [PY, "scripts/sft.py", "--config", p["sft_cfg"],
                           "--init-from", ckpt, "--chat-jsonl", ",".join(mix)])
    instruct_ckpt = find_latest(ROOT / p["sft_ckpt_dir"])

    # 5. reasoning: CoT distillation on <think>...</think> data, from the instruct model.
    # Prefer real GSM8K CoT (data/reason.jsonl from prepare_reason.py); else placeholder.
    reason_data = (args.reason_data
                   or ("data/reason.jsonl" if (ROOT / "data/reason.jsonl").exists()
                       else "data/sample_reason.jsonl"))
    final_ckpt = instruct_ckpt
    if instruct_ckpt is not None:
        print(f"   reasoning CoT data: {reason_data}")
        run("5 reasoning (CoT)", [PY, "scripts/sft.py", "--config", p["sft_cfg"],
                                  "--init-from", str(instruct_ckpt),
                                  "--chat-jsonl", reason_data,
                                  "--ckpt-dir", p["reason_ckpt_dir"]])
        final_ckpt = find_latest(ROOT / p["reason_ckpt_dir"]) or instruct_ckpt

    # 6. FINAL eval on the fully-trained model (perplexity + MCQ; real benchmarks too
    # when we used real data). Compare against the base eval in stage 3.
    if final_ckpt is not None:
        fck = str(final_ckpt)
        run("6 final-eval", [PY, "scripts/evaluate.py", "--ckpt", fck, "--tokenizer", p["tok_path"]])
        if p.get("real_posttrain"):
            run("6b benchmarks", [PY, "scripts/benchmark.py", "--ckpt", fck,
                                  "--tokenizer", p["tok_path"],
                                  "--tasks", "hellaswag,openbookqa,gsm8k,bfcl", "--limit", "100"])

    # 7. quantize the base (size/quality report)
    run("7 quantize", [PY, "scripts/quantize.py", "--ckpt", ckpt, "--tokenizer", p["tok_path"]])

    # 8. final sample from the REASONING model (should emit <think>...</think>)
    sft_ckpt = final_ckpt
    if sft_ckpt:
        run("8 sample (reasoning)", [PY, "-c",
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
