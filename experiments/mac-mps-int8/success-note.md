# Mac MPS Int8 Smoke Success

This note records the local Apple Silicon smoke run that proved the customized
runtime path works on Mac.

Runner:

```bash
python run_int8.py \
  --height 512 \
  --width 512 \
  --preset V4_TURBO_12 \
  --seed 0 \
  --out int8_out.png
```

Observed result from the successful run before generated artefacts were cleaned:

```text
Stage 1: text encoder loaded        alloc=8.78 GB   driver=8.81 GB
Stage 1: text encoder freed         alloc=0.47 GB   driver=1.64 GB
Stage 2: conditional loaded         alloc=9.76 GB   driver=10.75 GB
Stage 2: unconditional loaded       alloc=19.05 GB  driver=19.36 GB
Stage 2: vae loaded                 alloc=19.22 GB  driver=19.37 GB
Stage 3: denoising 12 steps at 512x512
Stage 3 done in 646.3s
SAVED int8_out.png
```

Why this matters:

- MPS cannot run Ideogram4 weight-only FP8 dequant directly.
- The local runtime re-quantizes FP8 weights to per-row int8 for MPS-compatible
  dequantization.
- Staged loading keeps peak driver memory below a 24 GB unified-memory machine.

Generated images and logs are intentionally ignored by git. Keep prompt files,
commands, and measured run notes in the repo; store bulky outputs separately.
