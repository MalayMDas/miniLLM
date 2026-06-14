# Simple one-word commands. Works on Linux / macOS / git-bash (the cloud is Linux).
# Windows local users can run the underlying `python ...` commands directly (see README).
IMAGE ?= llm-from-scratch:dev
CONFIG ?= configs/model_tiny.yaml

.PHONY: setup tok smoke tb test compare docker-build docker-run sky-dev sky-job clean

setup:            ## install pinned deps + package (editable)
	pip install -r requirements.lock && pip install -e .

tok:              ## train the byte-level BPE tokenizer
	python scripts/train_tokenizer.py --config $(CONFIG)

smoke: tok        ## end-to-end smoke-train (logs to runs/)
	python scripts/smoke_train.py --config $(CONFIG)

tb:               ## open TensorBoard on the runs/ dir
	tensorboard --logdir runs

compare:          ## show UTF-8 vs byte-level BPE token counts
	python -m llmscratch.tokenizer.compare --bpe artifacts/tok.json

test:             ## run the test suite
	pytest -q

docker-build:     ## build the reproducible CUDA image
	docker build -t $(IMAGE) .

docker-run:       ## run the sanity job in the container (needs nvidia-container-toolkit)
	docker run --rm --gpus all $(IMAGE)

sky-dev:          ## launch an interactive SkyPilot dev cluster on a spot GPU
	sky launch -c llm infra/sky/train.yaml

sky-job:          ## launch a managed spot job with auto-recovery
	sky jobs launch -n llm-train infra/sky/train.yaml

clean:            ## remove local run artifacts (keeps code)
	rm -rf runs artifacts .pytest_cache **/__pycache__
