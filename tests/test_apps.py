import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.apps.rag import RAGPipeline, chunk_text
from llmscratch.apps.agent import run_agent
from llmscratch.tools import default_registry, format_tool_call


def test_rag_retrieves_relevant_chunk():
    rag = RAGPipeline()
    rag.add_documents([
        "Paris is the capital of France. The Eiffel Tower is in Paris.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The mitochondria is the powerhouse of the cell.",
    ])
    hits = rag.retrieve("What is the capital of France?", k=1)
    assert "Paris" in hits[0][0].text


def test_chunking_splits_long_text():
    chunks = chunk_text(" ".join(f"sentence number {i}." for i in range(20)), max_words=10)
    assert len(chunks) > 1


def test_agent_uses_tool_then_answers():
    # stub model: first turn emits a calculator call, second turn gives final answer
    turns = iter([
        "I should compute this. " + format_tool_call("calculator", {"expression": "6*7"}),
        "The answer is 42.",
    ])

    def complete(messages):
        return next(turns)

    res = run_agent("What is 6 times 7?", default_registry(), complete, max_steps=3)
    assert "42" in res.answer
    assert res.steps == 2
    # the tool observation should have been fed back into the transcript
    assert any(m["role"] == "tool" and "42" in m["content"] for m in res.transcript)


def test_agent_direct_answer_no_tool():
    res = run_agent("hi", default_registry(), lambda m: "Hello!", max_steps=3)
    assert res.answer == "Hello!" and res.steps == 1
