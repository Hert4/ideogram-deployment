#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
load_dotenv

"$(python_bin)" "$ROOT/scripts/check_env.py"
