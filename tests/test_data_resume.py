"""Data-position checkpointing: skip_blocks must make a resumed stream continue
exactly where it stopped (no re-seeing early data). Tested with a deterministic
fake doc iterator so no network/HF download is needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.data.hf_stream import PackedHFStream
from llmscratch.tokenizer import ByteTokenizer

DOCS = [f"document number {i} has several words in it for packing" for i in range(60)]


def _stream(skip):
    s = PackedHFStream(ByteTokenizer(), "fake", block_size=8, skip_blocks=skip)
    s._doc_iter = lambda: iter(DOCS)        # bypass HF/network, deterministic order
    return [(x.tolist(), y.tolist()) for x, y in s]


def test_skip_continues_through_corpus():
    full = _stream(0)
    assert len(full) > 10
    skip_n = 5
    resumed = _stream(skip_n)
    # the resumed stream must equal the tail of the full stream after skip_n blocks
    assert resumed == full[skip_n:]


def test_skip_zero_is_noop():
    assert _stream(0) == _stream(0)


def test_skip_beyond_end_yields_nothing():
    full = _stream(0)
    assert _stream(len(full) + 100) == []
