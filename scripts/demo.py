"""One-command local demo: exercise every stage on CPU in a few seconds.

    python scripts/demo.py

No GPU, no downloads, no trained checkpoint required. Each section prints what it
did so you can SEE the pipeline work end to end. This is the fastest way to verify
a fresh clone is healthy (complements `pytest`, which checks invariants).

Note: the text model here is a tiny *untrained* toy, so generations are gibberish —
the point is that the mechanics (tokenization, scoring, tool calls, retrieval,
multimodal fusion, quantization) all run correctly.
"""
from __future__ import annotations

from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.tokenizer import ByteTokenizer, BPETokenizer
from llmscratch.model import Decoder, ModelConfig
from llmscratch.data import iter_local_lines


def header(title: str) -> None:
    print("\n" + "=" * 60 + f"\n  {title}\n" + "=" * 60)


def tiny_model(tok):
    torch.manual_seed(0)
    return Decoder(ModelConfig(vocab_size=tok.vocab_size, dim=64, n_layers=2,
                               n_heads=4, n_kv_heads=2, max_seq_len=128)).eval()


def demo_tokenizer():
    header("1. TOKENIZER  (byte UTF-8 vs trained byte-level BPE)")
    bpe = BPETokenizer.train(iter_local_lines("data/sample.txt"),
                             vocab_size=2000, min_frequency=1)
    byte = ByteTokenizer()
    text = "Large language models predict the next token."
    print(f"  text: {text!r}")
    print(f"  UTF-8 bytes : {len(byte.encode(text)):3d} tokens")
    print(f"  byte-BPE    : {len(bpe.encode(text)):3d} tokens  (compression in action)")
    assert bpe.decode(bpe.encode(text)).strip() == text


def demo_eval():
    header("2. EVALUATION  (perplexity + multiple-choice accuracy)")
    from llmscratch.eval import perplexity, multiple_choice_accuracy
    tok = ByteTokenizer()
    model = tiny_model(tok)
    ppl = perplexity(model, tok, ["the cat sat on the mat", "hello world"])
    mcq = [{"question": "2+2=", "choices": ["4", "5"], "answer": 0}]
    acc = multiple_choice_accuracy(model, tok, mcq)
    print(f"  perplexity (untrained model): {ppl:.1f}")
    print(f"  multiple-choice accuracy    : {acc:.2f}")


def demo_tools():
    header("3. TOOL USE  (registry + <tool_call> parse + safe execute)")
    from llmscratch.tools import default_registry, format_tool_call, execute_tool_calls
    reg = default_registry()
    emitted = "Let me compute. " + format_tool_call("calculator", {"expression": "6 * 7"})
    print(f"  model emitted : {emitted}")
    results = execute_tool_calls(emitted, reg)
    print(f"  executed      : {results[0]['name']} -> {results[0]['result']}")


def demo_rag():
    header("4. RAG  (chunk -> embed -> retrieve)")
    from llmscratch.apps.rag import RAGPipeline
    rag = RAGPipeline()
    n = rag.add_documents([
        "Paris is the capital of France. The Eiffel Tower is in Paris.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
    ])
    hits = rag.retrieve("What is the capital of France?", k=1)
    print(f"  indexed {n} chunks")
    print(f"  query -> top hit: {hits[0][0].text!r}  (score {hits[0][1]:.2f})")


def demo_agent():
    header("5. AGENT  (ReAct loop: reason -> act -> observe -> answer)")
    from llmscratch.apps.agent import run_agent
    from llmscratch.tools import default_registry, format_tool_call
    # stub model (a real trained model would emit these): call tool, then answer
    turns = iter([format_tool_call("calculator", {"expression": "6*7"}),
                  "The answer is 42."])
    res = run_agent("What is 6 times 7?", default_registry(),
                    lambda msgs: next(turns), max_steps=3)
    print(f"  steps: {res.steps} | answer: {res.answer!r}")


def demo_vision():
    header("6. VISION  (from-scratch ViT + projector -> multimodal forward)")
    from llmscratch.vision import ViT, ViTConfig, MultimodalDecoder
    tok = ByteTokenizer()
    decoder = tiny_model(tok)
    vit = ViT(ViTConfig(image_size=16, patch_size=8, dim=64, depth=2, heads=2))
    mm = MultimodalDecoder(decoder, vit, image_token_id=tok.token_to_id("<image>"))
    img_id = tok.token_to_id("<image>")
    ids = torch.tensor([[img_id] * vit.num_tokens + tok.encode("describe this")])
    pixels = torch.randn(1, 3, 16, 16)
    logits, _ = mm(ids, pixels)
    print(f"  {vit.num_tokens} image patches spliced into <image> slots")
    print(f"  multimodal logits shape: {tuple(logits.shape)}")


def demo_quantize():
    header("7. QUANTIZATION  (fp32 -> int8 size/quality)")
    from llmscratch.quantize import quantize_dynamic_int8, serialized_size_bytes
    tok = ByteTokenizer()
    model = tiny_model(tok)
    q = quantize_dynamic_int8(model)
    a, b = serialized_size_bytes(model) / 1e6, serialized_size_bytes(q) / 1e6
    print(f"  fp32: {a:.2f} MB  ->  int8: {b:.2f} MB   ({a/b:.2f}x smaller)")


def main():
    demo_tokenizer()
    demo_eval()
    demo_tools()
    demo_rag()
    demo_agent()
    demo_vision()
    demo_quantize()
    print("\nAll stages ran locally. For training stages see scripts/pretrain.py & sft.py.")


if __name__ == "__main__":
    main()
