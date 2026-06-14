# Cloud / infra

Three ways to run, from simplest to most reproducible. Pick per situation — they
share the same pinned `requirements.lock`, so the environment is consistent.

## A. Quick local test (no cloud, no Docker)
```bash
make setup && make smoke      # or the python commands in the root README
```
Fast iteration on your own machine. This is the default for development.

## B. Rented GPU box, no Docker (fastest cloud start)
Spin up a pod on **RunPod / Lambda / vast.ai** using their prebuilt
**PyTorch + CUDA** template, then:
```bash
git clone <your-repo> && cd llm-from-scratch
bash infra/cloud_setup.sh     # pinned install + GPU sanity + smoke-train
```
Best for cheap, throwaway experiments. Reproducibility comes from the lockfile +
recording the base-image tag.

## C. Reproducible image (Docker) — for repeatable training & serving
```bash
make docker-build             # builds CUDA image from the Dockerfile
make docker-run               # runs the sanity job (needs nvidia-container-toolkit)
```
Use this when you want the *exact* same environment every time, for serving
(vLLM), or for Kubernetes/Vertex. The base image pins CUDA so there's no
"CPU-only torch" surprise.

## Launching: orchestrator or not?
**Recommendation: SkyPilot, optionally.** It's the middle ground — no cluster to
operate, picks the cheapest spot GPU, auto-recovers preempted jobs:
```bash
sky jobs launch -n llm-train infra/sky/train.yaml   # managed spot, auto-resume
sky down llm
```
Skip Kubeflow/Airflow/Kubernetes unless you're running *many* pipelines across
*multiple teams* — they add an operational burden this project doesn't need.

## Reproducibility checklist (the cheap 80%)
- Pin deps: `requirements.lock` (regenerate with `pip freeze`).
- Record per run: git SHA + config hash + base-image tag (+ data hash later).
- Checkpoint to object storage (GCS/S3) every N steps → survive spot preemption.
