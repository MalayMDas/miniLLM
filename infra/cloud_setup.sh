#!/usr/bin/env bash
# No-Docker fast path: provision a rented GPU box (RunPod / Lambda / vast.ai) that
# already has a CUDA-enabled PyTorch base image. Run this from the repo root.
#
#   bash infra/cloud_setup.sh
#
# It pins deps from the lockfile, installs the package, and proves the GPU works
# with a sanity smoke-train before you spend money on a real run.
set -euo pipefail

echo "==> Python: $(python --version)"
echo "==> Installing pinned deps (requirements.lock) ..."
pip install --no-cache-dir -r requirements.lock
pip install --no-cache-dir -e .

echo "==> GPU sanity check ..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("device count:", torch.cuda.device_count())
else:
    print("WARNING: no CUDA device visible — check the base image / --gpus flag.")
PY

echo "==> Sanity smoke-train (tiny config) ..."
python scripts/train_tokenizer.py --config configs/model_tiny.yaml
python scripts/smoke_train.py     --config configs/model_tiny.yaml

echo "==> Done. For a real run, point --config at a larger config and"
echo "    launch under a process manager or 'sky jobs launch' (see infra/sky/train.yaml)."
