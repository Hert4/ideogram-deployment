#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
load_dotenv

: "${IDEOGRAM_DEVICE:=cuda}"
: "${IDEOGRAM_QUANTIZATION:=fp8}"
: "${IDEOGRAM_HEIGHT:=512}"
: "${IDEOGRAM_WIDTH:=512}"
: "${IDEOGRAM_PRESET:=V4_TURBO_12}"
: "${IDEOGRAM_SEED:=0}"
: "${IDEOGRAM_OUTPUT:=outputs/cuda_fp8_smoke.png}"

CAPTION="${IDEOGRAM_CAPTION:-$(caption_json)}"
OUTPUT_PATH="$(resolve_output_path "$IDEOGRAM_OUTPUT")"
prepare_output_dir "$OUTPUT_PATH"
PYTHON_BIN="$(python_bin)"

cd "$ROOT"
"$PYTHON_BIN" scripts/check_env.py

cd "$ROOT"
"$PYTHON_BIN" run_inference.py \
  --prompt "$CAPTION" \
  --no-magic-prompt \
  --warn-on-caption-issues \
  --quantization "$IDEOGRAM_QUANTIZATION" \
  --device "$IDEOGRAM_DEVICE" \
  --height "$IDEOGRAM_HEIGHT" \
  --width "$IDEOGRAM_WIDTH" \
  --sampler-preset "$IDEOGRAM_PRESET" \
  --seed "$IDEOGRAM_SEED" \
  --output "$OUTPUT_PATH"
