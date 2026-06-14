"""Parse tool calls the model emits, and format results back into the chat.

Convention (taught during SFT): the model emits
    <tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>
We extract + validate the JSON, run the tool via a ToolRegistry, and feed the
result back as a tool message for the next turn.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry

_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls = []
    for m in _TOOL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if "name" in obj:
                calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
        except json.JSONDecodeError:
            continue
    return calls


def format_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


def execute_tool_calls(text: str, registry: ToolRegistry) -> List[Dict[str, Any]]:
    """Run every tool call found in `text`; return [{name, arguments, result|error}]."""
    results = []
    for call in parse_tool_calls(text):
        try:
            res = registry.call(call["name"], call["arguments"])
            results.append({**call, "result": res})
        except Exception as e:  # surface tool errors to the model rather than crashing
            results.append({**call, "error": f"{type(e).__name__}: {e}"})
    return results


def tools_system_prompt(registry: ToolRegistry) -> str:
    """System message describing available tools + the call format."""
    schemas = json.dumps(registry.schemas(), indent=2)
    return ("You can call tools. Available tools:\n" + schemas +
            '\nTo call a tool, emit: <tool_call>{"name": "...", "arguments": {...}}'
            "</tool_call>")
