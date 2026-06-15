"""Fetch a real function-calling dataset as <tool_call> chat jsonl for the SFT mix.

    python scripts/prepare_tools.py                 # xLAM-60k -> data/tools.jsonl

Maps Salesforce/xlam-function-calling-60k ({query, tools, answers}) to:
    system: available tools (JSON) + call-format instruction
    user:   the query
    assistant: <tool_call>{"name":..,"arguments":{..}}</tool_call> for each gold call
so the model learns to read available tools and emit calls our parser understands.
Long tool schemas are fine; the SFT max_len truncates (raise it for real runs).
"""
from __future__ import annotations

try:
    import datasets  # noqa: F401
except Exception:
    pass

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.tools import format_tool_call


def _norm_name(name):
    if name is None or str(name).strip().lower() in ("", "none", "null"):
        return None
    return name


def _loads(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return None
    return x


def to_messages(query, tools, answers) -> dict | None:
    calls = _loads(answers)
    if not calls:
        return None
    parts = []
    for c in calls:
        if "name" in c:
            parts.append(format_tool_call(c["name"], c.get("arguments", c.get("parameters", {}))))
    if not parts:
        return None
    tools_str = tools if isinstance(tools, str) else json.dumps(_loads(tools) or tools)
    system = ("You can call tools. Available tools:\n" + tools_str +
              '\nTo call a tool, emit <tool_call>{"name": "...", "arguments": {...}}</tool_call>.')
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": query},
                         {"role": "assistant", "content": "".join(parts)}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Salesforce/xlam-function-calling-60k")
    ap.add_argument("--name", default="none")
    ap.add_argument("--split", default="train")
    ap.add_argument("--query-field", default="query")
    ap.add_argument("--tools-field", default="tools")
    ap.add_argument("--answers-field", default="answers")
    ap.add_argument("--max", type=int, default=10000)
    ap.add_argument("--out", default="data/tools.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, name=_norm_name(args.name), split=args.split,
                      streaming=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, skipped = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            msg = to_messages(ex.get(args.query_field, ""), ex.get(args.tools_field, ""),
                              ex.get(args.answers_field, ""))
            if msg is None:
                skipped += 1
                continue
            f.write(json.dumps(msg) + "\n")
            n += 1
            if n >= args.max:
                break
    print(f"wrote {n} tool-call conversations -> {out} (skipped {skipped}). "
          f"run_all folds it into the instruct SFT mix.")


if __name__ == "__main__":
    main()
