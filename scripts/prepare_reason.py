"""Fetch real chain-of-thought data and write it as <think>...</think> chat jsonl.

    python scripts/prepare_reason.py                       # GSM8K -> data/reason.jsonl
    python scripts/prepare_reason.py --max 5000 --out data/reason.jsonl

Default source is GSM8K (`gsm8k`/`main`): human-written step-by-step math solutions
ending in `#### <answer>`. We split the steps (the reasoning) from the final number
and emit:
    {"messages": [{"role": "user", "content": <question>},
                  {"role": "assistant", "content": "<think>{steps}</think>The answer is {n}."}]}

This matches the format the reasoning SFT stage trains on, so the model learns to
emit a <think> block then an answer. Other CoT sets with a {question, answer} schema
work via --question-field/--answer-field; sets that already contain <think> traces
(e.g. open-r1) can be passed through with --passthrough.
"""
from __future__ import annotations

# datasets/pyarrow before anything else (no torch here, but keep the habit).
try:
    import datasets  # noqa: F401
except Exception:
    pass

import argparse
import json
from pathlib import Path


def _norm_name(name):
    if name is None or str(name).strip().lower() in ("", "none", "null"):
        return None
    return name


def to_cot_message(question: str, answer: str) -> dict:
    """Map a {question, answer-with-steps} pair to a <think>-wrapped chat example."""
    if "####" in answer:
        steps, final = answer.split("####", 1)
        steps, final = steps.strip(), final.strip().replace(",", "")
        assistant = f"<think>{steps}</think>The answer is {final}."
    else:
        # already a free-form CoT answer; keep as-is inside <think>… is wrong, so just
        # treat the whole thing as the answer with a light reasoning preface.
        assistant = answer.strip()
    return {"messages": [{"role": "user", "content": question.strip()},
                         {"role": "assistant", "content": assistant}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--name", default="main", help="config name; 'none' for none")
    ap.add_argument("--split", default="train")
    ap.add_argument("--question-field", default="question")
    ap.add_argument("--answer-field", default="answer")
    ap.add_argument("--max", type=int, default=4000, help="max examples")
    ap.add_argument("--out", default="data/reason.jsonl")
    ap.add_argument("--passthrough", action="store_true",
                    help="answer already contains the full CoT/<think>; don't reformat")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, name=_norm_name(args.name), split=args.split,
                      streaming=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            q, a = ex[args.question_field], ex[args.answer_field]
            if args.passthrough:
                msg = {"messages": [{"role": "user", "content": q.strip()},
                                    {"role": "assistant", "content": a.strip()}]}
            else:
                msg = to_cot_message(q, a)
            f.write(json.dumps(msg) + "\n")
            n += 1
            if n >= args.max:
                break
    print(f"wrote {n} CoT examples -> {out}. "
          f"Use it: python scripts/run_all.py ... --reason-data {out}")


if __name__ == "__main__":
    main()
