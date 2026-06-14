# Reproducible training/serving image.
#
# Base = official PyTorch image with CUDA + cuDNN already correct. This removes the
# #1 cloud gotcha (CPU-only torch / CUDA-driver mismatch). The tag's CUDA version
# (12.1 here) must be <= the host GPU driver's supported CUDA — 12.1 is broadly safe.
# Verify/adjust the tag at https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Fail fast, no .pyc, unbuffered logs (so training output streams in `docker logs`).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# 1) Deps first (cached layer — only rebuilt when the lock changes).
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# 2) Then the package source (changes often — kept in its own layer).
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/ ./data/
RUN pip install --no-cache-dir -e .

# Default: a self-contained sanity run (train tokenizer -> smoke-train). Override
# at run time for real jobs, e.g.:
#   docker run --gpus all <img> python scripts/smoke_train.py --config configs/model_tiny.yaml
CMD ["bash", "-lc", "python scripts/train_tokenizer.py --config configs/model_tiny.yaml && python scripts/smoke_train.py --config configs/model_tiny.yaml"]
