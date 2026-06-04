#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

import int8_mps
import ideogram4.pipeline_ideogram4 as pip

pip.swap_linears_to_fp8 = int8_mps.swap_linears_to_int8
pip.load_fp8_state_dict = int8_mps.load_int8_state_dict

from huggingface_hub import hf_hub_download
from ideogram4 import Ideogram4PipelineConfig, PRESETS
from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from ideogram4.pipeline_ideogram4 import Ideogram4Pipeline, _load_autoencoder, _load_qwen3_vl
from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals

REPO_ID = "ideogram-ai/ideogram-4-fp8"


def cached_snapshot(repo_id: str) -> Path | None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_cache = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref = repo_cache / "refs" / "main"
    if not ref.exists():
        return None
    snapshot = repo_cache / "snapshots" / ref.read_text(encoding="utf-8").strip()
    return snapshot if snapshot.exists() else None


def resolve_repo() -> str:
    explicit = os.environ.get("IDEOGRAM_WEIGHTS_REPO")
    if explicit:
        return explicit
    snapshot = cached_snapshot(REPO_ID)
    if snapshot:
        print(f"Using cached local snapshot: {snapshot}", flush=True)
        return str(snapshot)
    return REPO_ID


def resolve_file(repo_id: str, filename: str, **_: object) -> str:
    root = Path(repo_id)
    if root.exists():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)
    return hf_hub_download(repo_id=repo_id, filename=filename)


REPO = resolve_repo()
pip.hf_hub_download = resolve_file
int8_mps.hf_hub_download = resolve_file


def mps_mem(tag: str) -> None:
    a = torch.mps.current_allocated_memory() / 1e9
    d = torch.mps.driver_allocated_memory() / 1e9
    print(f"[mem] {tag:26s} alloc={a:5.2f} GB  driver={d:5.2f} GB", flush=True)


def free() -> None:
    gc.collect()
    torch.mps.empty_cache()


def read_caption(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return json.dumps(json.load(f), ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("caption_files", nargs="+", type=Path)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--preset", default="V4_TURBO_12", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/mac-mps-int8/vi-quick"))
    args = ap.parse_args()

    assert torch.backends.mps.is_available(), "MPS not available"
    device = torch.device("mps")
    dtype = torch.bfloat16
    cfg = Ideogram4PipelineConfig(weights_repo=REPO)
    tcfg = Ideogram4Config()
    preset = PRESETS[args.preset]
    prompts = [read_caption(path) for path in args.caption_files]
    names = [path.stem for path in args.caption_files]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Batch: {len(prompts)} prompt(s), {args.width}x{args.height}, {args.preset}", flush=True)
    print("Stage 1: loading text encoder (int8)...", flush=True)
    tokenizer, text_encoder = _load_qwen3_vl(
        cfg.weights_repo,
        device,
        dtype,
        tokenizer_subfolder=cfg.tokenizer_subfolder,
        text_encoder_subfolder=cfg.text_encoder_subfolder,
    )
    mps_mem("text encoder loaded")

    pipe = Ideogram4Pipeline(
        conditional_transformer=None,
        unconditional_transformer=None,
        text_encoder=text_encoder,
        text_tokenizer=tokenizer,
        autoencoder=None,
        config=cfg,
        device=device,
        dtype=dtype,
    )
    pipe._verify_prompts(prompts, raise_on_issues=False)

    inputs = pipe._build_inputs(prompts, height=args.height, width=args.width)
    batch_size = len(prompts)
    num_image_tokens = inputs["num_image_tokens"]
    grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
    max_text_tokens = inputs["max_text_tokens"]
    latent_dim = tcfg.in_channels

    print("Stage 1: encoding prompts...", flush=True)
    llm_features = pipe._encode_text(
        inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
    )
    neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
    neg_segment_ids = inputs["segment_ids"][:, max_text_tokens:]
    neg_indicator = inputs["indicator"][:, max_text_tokens:]
    neg_llm_features = torch.zeros(
        batch_size,
        num_image_tokens,
        llm_features.shape[-1],
        dtype=llm_features.dtype,
        device=device,
    )

    pipe.text_encoder = None
    del text_encoder, tokenizer
    free()
    mps_mem("text encoder freed")
    print(f"Stage 1 done in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    print("Stage 2: streaming conditional transformer (int8)...", flush=True)
    cond = int8_mps.stream_build_int8_transformer(
        REPO, cfg.conditional_index_filename, tcfg, Ideogram4Transformer, device, dtype
    )
    free()
    mps_mem("conditional loaded")

    print("Stage 2: streaming unconditional transformer (int8)...", flush=True)
    uncond = int8_mps.stream_build_int8_transformer(
        REPO, cfg.unconditional_index_filename, tcfg, Ideogram4Transformer, device, dtype
    )
    free()
    mps_mem("unconditional loaded")

    ae_path = resolve_file(repo_id=REPO, filename=cfg.autoencoder_filename)
    autoencoder = _load_autoencoder(ae_path, device, dtype)
    pipe.conditional_transformer = cond
    pipe.unconditional_transformer = uncond
    pipe.autoencoder = autoencoder
    mps_mem("vae loaded")
    print(f"Stage 2 done in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    num_steps = preset.num_steps
    schedule = get_schedule_for_resolution(
        (args.height, args.width), known_mean=preset.mu, std=preset.std
    )
    step_intervals = make_step_intervals(num_steps)
    gw_per_step = torch.as_tensor(preset.guidance_schedule, dtype=torch.float32, device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    z = torch.randn(
        batch_size,
        num_image_tokens,
        latent_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    text_z_padding = torch.zeros(
        batch_size,
        max_text_tokens,
        latent_dim,
        dtype=torch.float32,
        device=device,
    )

    print(f"Stage 3: denoising {num_steps} steps...", flush=True)
    for i in range(num_steps - 1, -1, -1):
        st = time.time()
        t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
        s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
        t = torch.full((batch_size,), t_val, dtype=torch.float32, device=device)

        pos_z = torch.cat([text_z_padding, z], dim=1)
        pos_out = cond(
            llm_features=llm_features,
            x=pos_z,
            t=t,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
        )
        pos_v = pos_out[:, max_text_tokens:]
        neg_v = uncond(
            llm_features=neg_llm_features,
            x=z,
            t=t,
            position_ids=neg_position_ids,
            segment_ids=neg_segment_ids,
            indicator=neg_indicator,
        )
        gw_i = gw_per_step[i]
        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
        z = z + v * (s_val - t_val)
        torch.mps.synchronize()
        print(f"  step {num_steps - i}/{num_steps}  {time.time() - st:.1f}s", flush=True)

    images = pipe._decode(z, grid_h=grid_h, grid_w=grid_w)
    for name, image in zip(names, images, strict=True):
        out = args.out_dir / f"{name}.png"
        image.save(out)
        print(f"SAVED {out}", flush=True)
    print(f"Stage 3 done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
