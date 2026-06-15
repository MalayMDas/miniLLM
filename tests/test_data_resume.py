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


def test_weighted_interleave_mixes_by_weight():
    from itertools import islice
    from llmscratch.data import weighted_interleave
    # weights govern the order/rate; real use consumes a PREFIX (the token budget),
    # where the ratio holds. Sources large enough not to exhaust within the prefix.
    a, b = ["a"] * 5000, ["b"] * 5000
    prefix = list(islice(weighted_interleave([iter(a), iter(b)], [0.8, 0.2], seed=0), 1000))
    frac_a = prefix.count("a") / len(prefix)
    assert 0.7 < frac_a < 0.9               # ~80% from source a over the prefix
    # exhausting one source keeps yielding from the other
    short = list(weighted_interleave([iter(["x", "x"]), iter(["y"] * 50)], [0.5, 0.5], seed=1))
    assert short.count("y") == 50 and short.count("x") == 2


def test_bin_dataset_yields_windows(tmp_path):
    import numpy as np
    from llmscratch.data.bin_data import BinDataset
    p = tmp_path / "toy.bin"
    np.arange(1000, dtype=np.uint16).tofile(p)
    ds = BinDataset(p, block_size=16, seed=0)
    it = iter(ds)
    for _ in range(5):                       # infinite stream of random windows
        x, y = next(it)
        assert x.shape == (16,) and y.shape == (16,)
        # y is x shifted by one within a contiguous window
        assert int(y[0]) == int(x[1])
