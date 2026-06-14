"""Check training status while a job runs — without opening TensorBoard.

    python scripts/status.py                       # latest run, last 10 log points
    python scripts/status.py --run pretrain_local --tail 20
    python scripts/status.py --watch               # refresh every few seconds

Reads runs/<run>/metrics.jsonl (the trainer flushes it every log step) and the
latest checkpoint. For graphs, use `tensorboard --logdir runs` instead.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def latest_run(runs_dir: Path) -> Path | None:
    candidates = [p for p in runs_dir.glob("*") if (p / "metrics.jsonl").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "metrics.jsonl").stat().st_mtime)


def show(run_dir: Path, tail: int) -> None:
    mfile = run_dir / "metrics.jsonl"
    lines = mfile.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    train = [r for r in rows if "train/loss" in r]
    print(f"\n=== {run_dir.name} === ({len(train)} logged points)")
    for r in train[-tail:]:
        step = r.get("step", "?")
        loss = r.get("train/loss", float("nan"))
        lr = r.get("train/lr", float("nan"))
        tps = r.get("perf/tokens_per_sec")
        extra = f" | {tps:6.0f} tok/s" if tps else ""
        print(f"  step {step:>7} | loss {loss:.4f} | lr {lr:.2e}{extra}")

    # latest checkpoint (any artifacts/ckpt* dir)
    ckpts = sorted(ROOT.glob("artifacts/ckpt*/step_*.pt"))
    if ckpts:
        newest = max(ckpts, key=lambda p: p.stat().st_mtime)
        print(f"  latest checkpoint: {newest.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run name under runs/ (default: most recent)")
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument("--watch", action="store_true", help="refresh every --interval seconds")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    runs = ROOT / "runs"

    def resolve():
        return (runs / args.run) if args.run else latest_run(runs)

    while True:
        run_dir = resolve()
        if run_dir is None or not (run_dir / "metrics.jsonl").exists():
            print("no metrics yet — has training started? (runs/<name>/metrics.jsonl)")
        else:
            show(run_dir, args.tail)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
