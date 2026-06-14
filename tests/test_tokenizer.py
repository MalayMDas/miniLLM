import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.tokenizer import ByteTokenizer, BPETokenizer


def test_byte_roundtrip():
    tok = ByteTokenizer()
    for s in ["hello world", "café €3.50", "日本語", "def f(): return 1"]:
        assert tok.decode(tok.encode(s)) == s


def test_byte_specials():
    tok = ByteTokenizer()
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    # byte ids stay below 256
    assert all(i < 256 for i in ids[1:-1])


def test_bpe_train_and_roundtrip(tmp_path):
    corpus = ["the quick brown fox", "language models predict tokens"] * 50
    tok = BPETokenizer.train(iter(corpus), vocab_size=300, min_frequency=1)
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.vocab_size == tok.vocab_size
    s = "the quick brown fox"
    assert tok2.decode(tok2.encode(s)).strip() == s
    # bpe should compress vs raw bytes on in-distribution text
    assert len(tok2.encode(s)) < len(s.encode("utf-8"))
