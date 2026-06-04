#!/usr/bin/env python
from __future__ import annotations

import os
import platform
import sys


def show(key: str, value: object) -> None:
    print(f"{key:28s} {value}")


def main() -> int:
    show("python", sys.version.split()[0])
    show("platform", platform.platform())
    show("HF_HOME", os.environ.get("HF_HOME", ""))
    show("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    show("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))

    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic script
        show("torch import", f"FAILED: {exc}")
        return 1

    show("torch", torch.__version__)
    show("torch cuda build", getattr(torch.version, "cuda", None))
    show("float8_e4m3fn attr", hasattr(torch, "float8_e4m3fn"))

    cuda_ok = torch.cuda.is_available()
    show("cuda available", cuda_ok)
    if cuda_ok:
        show("cuda device count", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            show(f"cuda:{idx} name", props.name)
            show(f"cuda:{idx} memory GiB", round(props.total_memory / 1024**3, 2))
            show(f"cuda:{idx} capability", f"{props.major}.{props.minor}")

    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    show("mps available", mps_ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
