# Ideogram4 Test Harness

This repo contains a trimmed Ideogram4 runtime plus local runners and
environment presets for testing across:

- Apple Silicon / MPS (`run_int8.py`, proven smoke path)
- Linux CUDA workstations (`run_inference.py`)
- Run:ai H200 test/prod deployments

The repo root is the single GitHub project: runtime package, local integration
code, smoke tests, and deployment helpers all live here.

## Original Model Attribution

This repository is a local runtime adaptation and experiment harness for
Ideogram 4. The original model, weights, architecture, inference code, and
license are by the **Ideogram Team**.

Please cite and link the original work when using this repo:

- Technical report / launch note: [Ideogram 4.0 Technical Details: Open model at the forefront of design](https://ideogram.ai/blog/ideogram-4.0/)
- Original code repository: [ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4)
- Hugging Face model collection: [ideogram-ai/ideogram-4](https://huggingface.co/collections/ideogram-ai/ideogram-4)
- FP8 weights used by the Mac runner: [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8)
- NF4 weights for CUDA 24 GB tests: [ideogram-ai/ideogram-4-nf4](https://huggingface.co/ideogram-ai/ideogram-4-nf4)
- Model license: [Ideogram 4 Non-Commercial Model Agreement](model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL)

Suggested citation:

```bibtex
@misc{ideogram2026ideogram4,
  title        = {Ideogram 4.0 Technical Details: Open model at the forefront of design},
  author       = {{Ideogram Team}},
  year         = {2026},
  howpublished = {\url{https://ideogram.ai/blog/ideogram-4.0/}},
  note         = {Open-weight text-to-image model, inference code, and weights}
}
```

This repo is not an official Ideogram product. It keeps the original license
files and adds local changes only for Mac MPS, CUDA, and Run:ai experiments.

## Layout

```text
configs/env/        Environment presets for Mac, CUDA workstation, and Run:ai
configs/prompts/    Structured JSON smoke prompts
deploy/runai/       Run:ai/Kubernetes workload templates
docker/             CUDA image template
experiments/        Notes from successful local experiments
scripts/            Smoke runners and environment diagnostics
src/ideogram4/      Trimmed Ideogram4 runtime package
run_inference.py    CUDA/NF4/FP8 inference entrypoint
crates/             Rust terminal wrapper
```

## Quick Decision Table

| Environment | First target | Script | Notes |
| --- | --- | --- | --- |
| Mac M-series | `fp8 -> int8` workaround | `run_int8.py` | MPS cannot dequant native `float8_e4m3fn`; keep this path for local smoke tests. |
| Ubuntu RTX 4070 | CUDA FP8 smoke only | `scripts/smoke_cuda_fp8.sh` | Likely VRAM-limited for the stock loader; use small resolution first. |
| 24 GB CUDA GPU | NF4 baseline | `scripts/smoke_cuda_nf4.sh` | Ideogram's NF4 checkpoint is the practical 24 GB path. |
| Run:ai H200 | FP8 benchmark | `scripts/smoke_runai_h200_fp8.sh` | Start with 48 GiB GPU memory for smoke, then request more for prod benchmarks. |

## Tested Machine

Mac smoke tests were run successfully on:

```text
Machine: Apple Silicon Mac
OS: macOS 26.5 arm64
Python: 3.11.15
PyTorch: 2.12.0
Backend: MPS
Model: ideogram-ai/ideogram-4-fp8
Local path: FP8 checkpoint re-quantized to int8 for MPS-compatible dequant
```

Successful 512x512 single-image smoke run:

```text
Preset: V4_TURBO_12
Steps: 12
Peak MPS driver memory: ~19.4 GB
Stage 3 denoise time: ~646s
Output: int8_out.png
```

Successful Vietnamese batch smoke run:

```text
Resolution: 256x256
Batch size: 3 prompts
Preset: V4_TURBO_12
Steps: 12
Peak MPS driver memory: ~19.4 GB
Outputs: outputs/mac-mps-int8/vi-quick/*.png
```

## Resource Requirements

| Target | Minimum to try | Recommended | Notes |
| --- | ---: | ---: | --- |
| Apple Silicon MPS | 24 GB unified memory | 32 GB+ unified memory | The int8 MPS runner peaked around 19.4 GB driver memory; close other heavy apps. |
| RTX 4070 / 4070 Ti | 12-16 GB VRAM | Smoke only | Useful for CUDA wiring tests, but stock FP8 is likely VRAM-limited. |
| Single CUDA GPU NF4 | 24 GB VRAM | 24-48 GB VRAM | Best practical path for 24 GB CUDA cards. |
| Run:ai H200 FP8 smoke | 48 GiB GPU memory | 80 GiB+ | First DevOps ask; use full H200 for production benchmark if available. |
| Run:ai H200 production | 80 GiB | Full H200 | Needed for 1024/2048 tests, allocator headroom, service wrapper, and concurrency. |

Disk/cache recommendation:

```text
Hugging Face cache: 200-300 GB persistent volume for Run:ai
Local disk: keep the model cache outside git; generated outputs are ignored
```

## Required Secrets

- `HF_TOKEN`: required after accepting the Hugging Face model gate.
- `IDEOGRAM_API_KEY`: optional; required only when using plain text prompts with magic prompt enabled.
- `HIVE_TEXT_MODERATION_KEY` and `HIVE_VISUAL_MODERATION_KEY`: optional but recommended for production safety screening.

## Local Setup

Install the runtime and smoke harness in editable mode:

```bash
uv sync
uv pip install -e .
```

If you are using plain `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Smoke Commands

Mac MPS int8 workaround:

```bash
scripts/smoke_mac_mps_int8.sh
```

Linux CUDA FP8:

```bash
scripts/smoke_cuda_fp8.sh
```

Linux CUDA NF4:

```bash
scripts/smoke_cuda_nf4.sh
```

Run:ai H200 FP8 command body:

```bash
scripts/smoke_runai_h200_fp8.sh
```

Environment diagnostics only:

```bash
scripts/check_env.sh
```

Run with one Vietnamese sample prompt:

```bash
IDEOGRAM_CAPTION="$(scripts/read_prompt.py configs/prompts/vi/ca-phe-misa-product-ad.json)" \
scripts/smoke_mac_mps_int8.sh
```

Run all three Vietnamese prompts on Mac MPS:

```bash
scripts/run_vi_prompts_mac.sh
```

## Rust Terminal Wrapper

The Rust wrapper is an orchestration layer: it does not port the model to Rust.
It provides a terminal interface and calls the existing Python/PyTorch runners.

Build:

```bash
cargo build -p ideogram4-cli
```

Check environment:

```bash
cargo run -p ideogram4-cli -- check
```

Generate from a structured JSON prompt:

```bash
cargo run -p ideogram4-cli -- generate \
  --prompt-file configs/prompts/vi/poster-tet-an-lanh.json \
  --backend mac-int8 \
  --size 256 256 \
  --output outputs/rust-chat/poster-tet-an-lanh.png
```

Generate from free text:

```bash
cargo run -p ideogram4-cli -- generate \
  --prompt "poster Tết chữ TẾT AN LÀNH" \
  --backend mac-int8 \
  --size 256 256
```

Interactive terminal:

```bash
cargo run -p ideogram4-cli -- chat --backend mac-int8 --size 256 256
```

Inside chat:

```text
/backend mac-int8
/size 256 256
/file configs/prompts/vi/ca-phe-misa-product-ad.json
poster Tết chữ TẾT AN LÀNH
/quit
```

## Environment Presets

Copy one of these files and fill in the secrets:

- `configs/env/mac-mps-int8.env.example`
- `configs/env/linux-cuda-4070-fp8.env.example`
- `configs/env/linux-cuda-24gb-nf4.env.example`
- `configs/env/runai-h200-fp8.env.example`

Example:

```bash
cp configs/env/runai-h200-fp8.env.example .env
set -a
source .env
set +a
```

## Run:ai First Ask

For the first FP8 smoke test on H200, request the smallest practical allocation:

```text
GPU: NVIDIA H200
GPU memory: 48GiB
CPU: 8-16 cores
RAM: 64-96GiB
PVC/cache: 200-300GiB for Hugging Face cache
```

If 48 GiB is unavailable, 32 GiB can be tried as a risky smoke target at
512x512. For production benchmarks, request 80 GiB or a full H200.

The initial workload template is:

```text
deploy/runai/ideogram4-h200-fp8-smoke.template.yaml
```

The CUDA image template is:

```text
docker/Dockerfile.cuda
```

## Runtime Notes

- `run_int8.py`: current Apple Silicon success path.
- `int8_mps.py`: local FP8-to-int8 loader for MPS compatibility.
- Failed/offline experiment logs and earlier runner attempts are intentionally not kept.

## Repo Hygiene

Generated images, logs, virtualenvs, and Hugging Face caches are ignored. Keep
only small scripts, configs, and notes in git.
