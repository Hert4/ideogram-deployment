#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: scripts/read_prompt.py <caption.json>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    with path.open("r", encoding="utf-8") as f:
        caption = json.load(f)
    print(json.dumps(caption, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
