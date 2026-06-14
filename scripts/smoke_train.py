"""End-to-end smoke train: prove the whole loop on one machine before cloud spend.

    python scripts/train_tokenizer.py --config configs/model_tiny.yaml   # if mode=bpe
    python scripts/smoke_train.py     --config configs/model_tiny.yaml

It overfits a tiny corpus: loss should fall fast and the sample should start to
echo the training text. That confirms tokenizer -> data -> model -> optimizer ->
generation are all wired correctly.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmscratch.data import iter_local_lines, encode_corpus, PackedDataset
from llmscratch.model import Decoder, ModelConfig
from llmscratch.tokenizer import build_tokenizer
from llmscratch.utils import build_logger


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * prog))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_tiny.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    t = cfg["train"]
    torch.manual_seed(t["seed"])

    device = pick_device(t["device"])
    print(f"device: {device}")

    tok = build_tokenizer(cfg["tokenizer"])
    print(f"tokenizer: {cfg['tokenizer']['mode']} (vocab_size={tok.vocab_size})")

    ids = encode_corpus(tok, iter_local_lines(cfg["data"]["corpus"]))
    ds = PackedDataset(ids, cfg["data"]["block_size"])
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, drop_last=True)
    print(f"corpus tokens: {len(ids)} | blocks: {len(ds)}")

    m = cfg["model"]
    model = Decoder(ModelConfig(
        vocab_size=tok.vocab_size,
        dim=m["dim"], n_layers=m["n_layers"], n_heads=m["n_heads"],
        n_kv_heads=m["n_kv_heads"], max_seq_len=m["max_seq_len"], dropout=m["dropout"],
    )).to(device)
    print(f"model params: {model.num_params()/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                            weight_decay=t["weight_decay"], betas=(0.9, 0.95))

    logger = build_logger(cfg.get("logging", {}), cfg["logging"].get("run_name", "smoke"), cfg)
    logger.log_hparams({"model": m, "train": t, "tokenizer": cfg["tokenizer"]})
    tokens_per_step = t["batch_size"] * cfg["data"]["block_size"]

    def sample(prompt: str = "The quick brown") -> str:
        model.eval()
        start = torch.tensor([[tok.bos_id] + tok.encode(prompt)], device=device)
        out = model.generate(start, max_new_tokens=40, temperature=0.8, top_k=40)
        model.train()
        return tok.decode(out[0].tolist())

    model.train()
    step = 0
    it = iter(dl)
    while step < t["steps"]:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        lr = lr_at(step, t["lr"], t["warmup_steps"], t["steps"])
        for g in opt.param_groups:
            g["lr"] = lr

        t0 = time.perf_counter()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        logger.log_scalars({
            "train/loss": loss.item(),
            "train/perplexity": math.exp(min(loss.item(), 20)),
            "train/lr": lr,
            "train/grad_norm": float(gnorm),
            "perf/tokens_per_sec": tokens_per_step / max(dt, 1e-6),
            "perf/step_time_ms": dt * 1e3,
        }, step)

        if step % t.get("sample_every", 50) == 0:
            logger.log_text("samples/generation", sample(), step)
        if step % 20 == 0 or step == t["steps"] - 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"{tokens_per_step / max(dt, 1e-6):.0f} tok/s")
        step += 1

    final = sample()
    logger.log_text("samples/generation", final, step)
    logger.close()
    print("\n--- sample ---")
    print(final)
    print(f"\nlogged to: {Path(cfg['logging'].get('logdir','runs')) / cfg['logging'].get('run_name','smoke')}")
    print("view with:  tensorboard --logdir runs")


if __name__ == "__main__":
    main()
