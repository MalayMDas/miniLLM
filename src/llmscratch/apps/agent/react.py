"""A minimal ReAct-style agent loop (Reason + Act).

The model alternates between emitting a tool call and reading the result, until it
produces a final answer or hits the step limit. The loop is model-agnostic: pass a
`complete(messages) -> str` callable, so it works with our trained model, a stub
(for tests), or any API. This is the skeleton every agent framework (LangChain,
the Anthropic SDK agent loop, etc.) elaborates on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from ...tools import ToolRegistry, execute_tool_calls, parse_tool_calls, tools_system_prompt

Complete = Callable[[List[Dict[str, str]]], str]


@dataclass
class AgentResult:
    answer: str
    steps: int
    transcript: List[Dict[str, str]] = field(default_factory=list)


def run_agent(question: str, registry: ToolRegistry, complete: Complete,
              max_steps: int = 5) -> AgentResult:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": tools_system_prompt(registry)},
        {"role": "user", "content": question},
    ]
    for step in range(1, max_steps + 1):
        out = complete(messages)
        messages.append({"role": "assistant", "content": out})

        calls = parse_tool_calls(out)
        if not calls:
            return AgentResult(answer=out.strip(), steps=step, transcript=messages)

        # execute every requested tool and feed observations back
        results = execute_tool_calls(out, registry)
        obs = "\n".join(
            f"{r['name']} -> {r.get('result', r.get('error'))}" for r in results)
        messages.append({"role": "tool", "content": obs})

    return AgentResult(answer="(max steps reached)", steps=max_steps, transcript=messages)
