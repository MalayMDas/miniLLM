"""Export a checkpoint to GGUF (for llama.cpp / Ollama / LM Studio — runs on CPU /
Apple Silicon / low-VRAM). This is the plan's primary "runs anywhere" deployment path.

Pipeline: our checkpoint -> HF Llama folder (export_hf, numerically exact) ->
llama.cpp's `convert_hf_to_gguf.py` -> optional `llama-quantize` to 4-bit.

    # 1) get llama.cpp once:  git clone https://github.com/ggerganov/llama.cpp
    # 2) convert + quantize:
    python scripts/export_gguf.py --ckpt artifacts/ckpt_sft_local/step_X.pt \
        --tokenizer artifacts/tok_local.json --llama-cpp ../llama.cpp \
        --out artifacts/model.gguf --quantize Q4_K_M

Then run it:  ../llama.cpp/build/bin/llama-cli -m artifacts/model.q4_k_m.gguf -p "Hello"

Without --llama-cpp it just produces the HF folder and prints the exact commands.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.model.hf_export import export_hf


def _find(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tok_local.json")
    ap.add_argument("--hf-dir", default="artifacts/hf_model")
    ap.add_argument("--out", default="artifacts/model.gguf")
    ap.add_argument("--llama-cpp", default=None, help="path to a cloned llama.cpp repo")
    ap.add_argument("--outtype", default="f16", help="f16 | bf16 | f32 (pre-quant precision)")
    ap.add_argument("--quantize", default=None, help="e.g. Q4_K_M, Q5_K_M, Q8_0 (optional)")
    args = ap.parse_args()

    # 1) HF Llama folder (the GGUF converter consumes this).
    export_hf(args.ckpt, args.hf_dir, tokenizer=args.tokenizer)

    if not args.llama_cpp:
        print("\n[no --llama-cpp given] HF folder is ready. To finish:")
        print("  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && pip install -r requirements.txt")
        print(f"  python convert_hf_to_gguf.py {args.hf_dir} --outfile {args.out} --outtype {args.outtype}")
        print("  cmake -B build && cmake --build build --config Release   # builds llama-quantize/llama-cli")
        print(f"  build/bin/llama-quantize {args.out} model.q4_k_m.gguf Q4_K_M")
        return

    lcpp = Path(args.llama_cpp)
    convert = lcpp / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit(f"convert_hf_to_gguf.py not found under {lcpp}")

    # 2) convert HF -> GGUF
    print(f"\nconverting -> {args.out} ...")
    subprocess.run([sys.executable, str(convert), args.hf_dir,
                    "--outfile", args.out, "--outtype", args.outtype], check=True)

    # 3) optional quantize
    if args.quantize:
        qbin = _find([lcpp / "build/bin/llama-quantize", lcpp / "build/bin/llama-quantize.exe",
                      lcpp / "llama-quantize", lcpp / "build/bin/quantize"])
        if qbin is None:
            print("built llama-quantize not found; build llama.cpp (cmake --build build) then run:")
            print(f"  <llama.cpp>/build/bin/llama-quantize {args.out} "
                  f"{Path(args.out).with_suffix('')}.{args.quantize.lower()}.gguf {args.quantize}")
            return
        qout = str(Path(args.out).with_suffix("")) + f".{args.quantize.lower()}.gguf"
        subprocess.run([str(qbin), args.out, qout, args.quantize], check=True)
        print(f"quantized -> {qout}")


if __name__ == "__main__":
    main()
