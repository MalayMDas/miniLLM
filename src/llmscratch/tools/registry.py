"""Tool registry + a few safe built-in tools.

A "tool" is a name + JSON-schema of arguments + a Python callable. The schema is
what we (a) show the model in the system prompt and (b) validate calls against.
This mirrors function-calling in the Anthropic/OpenAI APIs.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]      # JSON-schema-ish: {arg: {type, description}}
    fn: Callable[..., Any]

    def schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        return self.get(name).fn(**arguments)


# ---- safe arithmetic (no eval(); whitelisted AST nodes) --------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression safely."""
    return _safe_eval(ast.parse(expression, mode="eval"))


def default_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(Tool(
        name="calculator",
        description="Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'.",
        parameters={"expression": {"type": "string", "description": "arithmetic expression"}},
        fn=calculator,
    ))
    return r
