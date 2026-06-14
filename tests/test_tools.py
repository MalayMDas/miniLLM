import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.tools import (default_registry, calculator, parse_tool_calls,
                              format_tool_call, execute_tool_calls)


def test_calculator_safe():
    assert calculator("2 * (3 + 4)") == 14
    assert calculator("2 ** 10") == 1024


def test_calculator_rejects_code():
    import pytest
    with pytest.raises(ValueError):
        calculator("__import__('os').system('echo hi')")


def test_parse_and_format_roundtrip():
    s = format_tool_call("calculator", {"expression": "1+1"})
    calls = parse_tool_calls("sure! " + s + " done")
    assert calls == [{"name": "calculator", "arguments": {"expression": "1+1"}}]


def test_execute_tool_calls():
    reg = default_registry()
    text = format_tool_call("calculator", {"expression": "6*7"})
    results = execute_tool_calls(text, reg)
    assert results[0]["result"] == 42


def test_execute_reports_error_not_crash():
    reg = default_registry()
    text = format_tool_call("calculator", {"expression": "1/0"})
    results = execute_tool_calls(text, reg)
    assert "error" in results[0]
