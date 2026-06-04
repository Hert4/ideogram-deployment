#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
load_dotenv

: "${IDEOGRAM_HEIGHT:=512}"
: "${IDEOGRAM_WIDTH:=512}"
: "${IDEOGRAM_PRESET:=V4_TURBO_12}"
: "${IDEOGRAM_SEED:=0}"

PROMPTS=(
  "ca-phe-misa-product-ad"
  "poster-tet-an-lanh"
  "bien-hieu-pho-dem-ha-noi"
)

for name in "${PROMPTS[@]}"; do
  prompt_path="$ROOT/configs/prompts/vi/${name}.json"
  output_path="$ROOT/outputs/mac-mps-int8/${name}.png"
  mkdir -p "$(dirname "$output_path")"

  echo "Running $name -> $output_path"
  IDEOGRAM_CAPTION="$("$ROOT/scripts/read_prompt.py" "$prompt_path")" \
  IDEOGRAM_HEIGHT="$IDEOGRAM_HEIGHT" \
  IDEOGRAM_WIDTH="$IDEOGRAM_WIDTH" \
  IDEOGRAM_PRESET="$IDEOGRAM_PRESET" \
  IDEOGRAM_SEED="$IDEOGRAM_SEED" \
  IDEOGRAM_OUTPUT="$output_path" \
  "$ROOT/scripts/smoke_mac_mps_int8.sh"
done
