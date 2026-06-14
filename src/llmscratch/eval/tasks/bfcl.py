"""BFCL-style function-calling eval — scored by AST/argument match (local, no API).

The Berkeley Function-Calling Leaderboard has several categories; the "executable"
ones run real APIs, but the **AST categories** (the bulk) are scored by comparing the
predicted call's function name + arguments to the gold call — fully local. We
implement that match here and run it on a jsonl of {question, tools, gold}.

Match rule: same function name AND every gold argument present with an equal value
(string-normalized). Extra optional args are tolerated unless you tighten it.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from ...tools import parse_tool_calls, tools_system_prompt, ToolRegistry, Tool


def _norm(v) -> str:
    return str(v).strip().lower()


def call_matches(pred: Optional[Dict], gold: Dict) -> bool:
    if pred is None:
        return False
    if pred.get("name") != gold.get("name"):
        return False
    gold_args = gold.get("arguments", {})
    pred_args = pred.get("arguments", {})
    return all(k in pred_args and _norm(pred_args[k]) == _norm(v)
               for k, v in gold_args.items())


def load_bfcl_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Each line: {question, tools: [schema...], gold: {name, arguments}}."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def _registry_from_schemas(schemas: List[Dict]) -> ToolRegistry:
    reg = ToolRegistry()
    for s in schemas:
        reg.register(Tool(name=s["name"], description=s.get("description", ""),
                          parameters=s.get("parameters", {}), fn=lambda **kw: None))
    return reg


def evaluate_bfcl(model, tokenizer, examples: List[Dict], device: str = "cpu",
                  max_new_tokens: int = 64) -> float:
    from ...serve.generate import generate
    correct = 0
    for ex in examples:
        reg = _registry_from_schemas(ex["tools"])
        prompt = tools_system_prompt(reg) + "\n\nUser: " + ex["question"] + "\nAssistant:"
        ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
        out = generate(model, ids, max_new_tokens=max_new_tokens, temperature=0.0,
                       top_k=None, top_p=None, stop_ids=[tokenizer.eos_id], device=device)
        calls = parse_tool_calls(tokenizer.decode(out))
        correct += int(call_matches(calls[0] if calls else None, ex["gold"]))
    return correct / max(len(examples), 1)
