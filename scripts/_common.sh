#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

load_dotenv() {
  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
}

python_bin() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$ROOT/.venv/bin/python"
  else
    command -v python
  fi
}

caption_json() {
  "$(python_bin)" - "$ROOT/configs/prompts/smoke_caption.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.dumps(json.load(f), separators=(",", ":")))
PY
}

prepare_output_dir() {
  local output_path="$1"
  local output_dir
  output_dir="$(dirname "$output_path")"
  if [[ "$output_dir" != "." ]]; then
    mkdir -p "$output_dir"
  fi
}

resolve_output_path() {
  local output_path="$1"
  if [[ "$output_path" = /* ]]; then
    printf '%s\n' "$output_path"
  else
    printf '%s/%s\n' "$ROOT" "$output_path"
  fi
}
