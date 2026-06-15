"""Fetch a real instruction-tuning dataset as chat jsonl for the SFT stage.

    python scripts/prepare_instruct.py                      # UltraChat-200k -> data/instruct.jsonl
    python scripts/prepare_instruct.py --dataset teknium/OpenHermes-2.5 --name none --split train --max 20000

Auto-detects the schema:
  - `messages` (UltraChat): [{role, content}, ...] -> passed through (role-normalized)
  - `conversations` (ShareGPT / OpenHermes): [{from, value}] -> mapped (human->user, gpt->assistant)
Writes {"messages":[...]} lines that the SFT stage consumes. Long convs are fine;
the SFT max_len truncates them.
"""
from __future__ import annotations

try:
    import datasets  # noqa: F401  (pyarrow before torch isn't needed here; no torch)
except Exception:
    pass

import argparse
import json
from pathlib import Path

_ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
         "system": "system", "tool": "tool", "function": "tool"}


def _norm_name(name):
    if name is None or str(name).strip().lower() in ("", "none", "null"):
        return None
    return name


def to_messages(ex) -> list | None:
    if "messages" in ex and ex["messages"]:
        out = []
        for m in ex["messages"]:
            r = _ROLE.get(str(m.get("role", "")).lower())
            if r and m.get("content"):
                out.append({"role": r, "content": m["content"]})
        return out or None
    if "conversations" in ex and ex["conversations"]:
        out = []
        for c in ex["conversations"]:
            r = _ROLE.get(str(c.get("from", "")).lower())
            if r and c.get("value"):
                out.append({"role": r, "content": c["value"]})
        return out or None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--name", default="none")
    ap.add_argument("--split", default="train_sft")
    ap.add_argument("--max", type=int, default=20000)
    ap.add_argument("--out", default="data/instruct.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, name=_norm_name(args.name), split=args.split,
                      streaming=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, skipped = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            msgs = to_messages(ex)
            if not msgs or not any(m["role"] == "assistant" for m in msgs):
                skipped += 1
                continue
            f.write(json.dumps({"messages": msgs}) + "\n")
            n += 1
            if n >= args.max:
                break
    print(f"wrote {n} instruct conversations -> {out} (skipped {skipped} malformed). "
          f"run_all auto-uses it for the instruct SFT stage.")


if __name__ == "__main__":
    main()
