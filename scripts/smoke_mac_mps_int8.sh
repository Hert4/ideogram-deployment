#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
load_dotenv

: "${IDEOGRAM_HEIGHT:=512}"
: "${IDEOGRAM_WIDTH:=512}"
: "${IDEOGRAM_PRESET:=V4_TURBO_12}"
: "${IDEOGRAM_SEED:=0}"
: "${IDEOGRAM_OUTPUT:=int8_out.png}"

CAPTION="${IDEOGRAM_CAPTION:-$(caption_json)}"
OUTPUT_PATH="$(resolve_output_path "$IDEOGRAM_OUTPUT")"
prepare_output_dir "$OUTPUT_PATH"
PYTHON_BIN="$(python_bin)"

cd "$ROOT"
"$PYTHON_BIN" scripts/check_env.py
"$PYTHON_BIN" run_int8.py \
  --caption "$CAPTION" \
  --height "$IDEOGRAM_HEIGHT" \
  --width "$IDEOGRAM_WIDTH" \
  --preset "$IDEOGRAM_PRESET" \
  --seed "$IDEOGRAM_SEED" \
  --out "$OUTPUT_PATH"
