"""Generate tiled-refine workflow vectors with the pinned ComfyUI extension.

Usage from the Dinkster repository root::

    .venv-torch/bin/python tools/gen_usdu_golden.py \
        /path/to/ComfyUI-at-a1079ba1 \
        /path/to/ComfyUI_UltimateSDUpscale-at-a5547db9 \
        /path/to/v1-5-pruned-emaonly-fp16.safetensors \
        /path/to/RealESRGAN_x4plus_anime_6B.pth

Use Python 3.12 with the package versions recorded in the generated payload.
The extension checkout must include its initialized ultimate_sd_upscale submodule.

Artifact provenance (verified against the byte counts and SHA-256 pins below):

- v1-5-pruned-emaonly-fp16.safetensors (2,132,696,762 bytes) from
  https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/9cfd069101959ca3828bf9c04a4419870832b74f/v1-5-pruned-emaonly-fp16.safetensors
- RealESRGAN_x4plus_anime_6B.pth (17,938,799 bytes) from
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
from PIL import Image, ImageFilter

COMFYUI_COMMIT = "a1079ba16f2674734b065eb036fbfdddaa321a4d"
EXTENSION_COMMIT = "a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df"
UPSTREAM_COMMIT = "2322caa480535b1011a1f9c18126d85ea444f146"
ARTIFACTS = {
    "checkpoint": {
        "bytes": 2_132_696_762,
        "sha256": "e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916",
        "url": (
            "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/"
            "9cfd069101959ca3828bf9c04a4419870832b74f/"
            "v1-5-pruned-emaonly-fp16.safetensors"
        ),
    },
    "upscale_model": {
        "bytes": 17_938_799,
        "sha256": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/"
            "RealESRGAN_x4plus_anime_6B.pth"
        ),
    },
}
CASES = {
    "linear": {"mode_type": "Linear", "seam_fix_mode": "None"},
    "band_pass": {"mode_type": "Linear", "seam_fix_mode": "Band Pass"},
}
PARAMETERS = {
    "positive_text": "high detail photograph",
    "negative_text": "",
    "upscale_by": 2.0,
    "seed": 123,
    "steps": 1,
    "cfg": 8.0,
    "sampler_name": "euler",
    "scheduler": "simple",
    "denoise": 0.2,
    "tile_width": 64,
    "tile_height": 64,
    "mask_blur": 8,
    "tile_padding": 32,
    "seam_fix_denoise": 1.0,
    "seam_fix_width": 64,
    "seam_fix_mask_blur": 8,
    "seam_fix_padding": 16,
    "force_uniform_tiles": True,
    "tiled_decode": False,
    "batch_size": 1,
}
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "usdu_a5547db9.json"


def _source() -> np.ndarray:
    height, width = 32, 64
    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = ((rows * 11 + columns * 3) % 256).astype(np.uint8)
    frame[..., 1] = ((rows * 5 + columns * 13) % 256).astype(np.uint8)
    frame[..., 2] = ((rows * 17 + columns * 7) % 256).astype(np.uint8)
    frame[5:14, 7:21] = (240, 32, 96)
    frame[19:29, 39:57] = (24, 224, 160)
    return frame


def _record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": list(contiguous.shape),
        "uint8Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _check_checkout(path: Path, expected: str, label: str) -> None:
    head = _git(path, "rev-parse", "HEAD")
    if head != expected:
        raise SystemExit(f"{label} must be checked out at {expected}, got {head}")
    if _git(path, "status", "--porcelain"):
        raise SystemExit(f"{label} checkout must be clean")


def _check_artifact(path: Path, expected: dict[str, object], label: str) -> None:
    if path.stat().st_size != expected["bytes"]:
        raise SystemExit(f"{label} byte count does not match the pinned artifact")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise SystemExit(f"{label} SHA-256 does not match the pinned artifact")


def _node_result(result: object, index: int = 0) -> Any:
    values = getattr(result, "result", result)
    if not isinstance(values, (list, tuple)):
        raise TypeError("node result must be a list or tuple")
    return values[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comfyui", type=Path)
    parser.add_argument("extension", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("upscale_model", type=Path)
    args = parser.parse_args()
    comfyui = args.comfyui.resolve()
    extension = args.extension.resolve()
    checkpoint = args.checkpoint.resolve()
    upscale_model = args.upscale_model.resolve()
    _check_checkout(comfyui, COMFYUI_COMMIT, "ComfyUI")
    _check_checkout(extension, EXTENSION_COMMIT, "UltimateSDUpscale")
    _check_checkout(
        extension / "repositories" / "ultimate_sd_upscale",
        UPSTREAM_COMMIT,
        "ultimate_sd_upscale submodule",
    )
    _check_artifact(checkpoint, ARTIFACTS["checkpoint"], "checkpoint")
    _check_artifact(upscale_model, ARTIFACTS["upscale_model"], "upscale model")

    sys.argv = [sys.argv[0], "--cpu"]
    sys.path.insert(0, str(comfyui))
    sys.path.insert(0, str(extension))
    import comfy.options  # pyright: ignore[reportMissingImports]

    comfy.options.enable_args_parsing()
    import folder_paths  # pyright: ignore[reportMissingImports]
    import nodes  # pyright: ignore[reportMissingImports]
    import usdu_nodes  # pyright: ignore[reportMissingImports]
    from comfy_extras.nodes_upscale_model import (  # pyright: ignore[reportMissingImports]
        UpscaleModelLoader,
    )
    from modules import processing as usdu_processing  # pyright: ignore[reportMissingImports]

    folder_paths.add_model_folder_path("checkpoints", str(checkpoint.parent), is_default=True)
    folder_paths.add_model_folder_path("upscale_models", str(upscale_model.parent), is_default=True)

    torch.set_grad_enabled(False)
    model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint.name)
    positive = _node_result(nodes.CLIPTextEncode().encode(clip, PARAMETERS["positive_text"]))
    negative = _node_result(nodes.CLIPTextEncode().encode(clip, PARAMETERS["negative_text"]))
    upscaler = _node_result(UpscaleModelLoader().load_model(upscale_model.name))
    source = _source()
    image = torch.from_numpy(source[None].astype(np.float32) / 255.0)
    outputs: dict[str, object] = {}
    jobs: dict[str, object] = {}
    original_process_images = usdu_processing.process_images
    active_jobs: list[dict[str, object]] = []

    def capture_process_images(process: Any) -> Any:
        before = process.init_images[0].copy()
        mask = process.image_mask.copy()
        crop_region = usdu_processing.get_crop_region(mask, process.inpaint_full_res_padding)
        x1, y1, x2, y2 = crop_region
        crop_width, crop_height = x2 - x1, y2 - y1
        if process.uniform_tile_mode:
            crop_ratio = crop_width / crop_height if crop_height else 1.0
            sample_ratio = process.width / process.height if process.height else 1.0
            if crop_ratio > sample_ratio:
                target_width = crop_width
                target_height = round(crop_width / sample_ratio)
            else:
                target_width = round(crop_height * sample_ratio)
                target_height = crop_height
            crop_region, _ = usdu_processing.expand_crop(
                crop_region,
                mask.width,
                mask.height,
                target_width,
                target_height,
            )
        else:
            target_width = ((crop_width + 7) // 8) * 8
            target_height = ((crop_height + 7) // 8) * 8
            crop_region, _ = usdu_processing.expand_crop(
                crop_region,
                mask.width,
                mask.height,
                target_width,
                target_height,
            )
        sampled: list[Any] = []
        original_tensor_to_pil = usdu_processing.tensor_to_pil

        def capture_tensor_to_pil(value: Any, index: int = 0) -> Any:
            tile = original_tensor_to_pil(value, index)
            sampled.append(tile.copy())
            return tile

        usdu_processing.tensor_to_pil = capture_tensor_to_pil
        try:
            result = original_process_images(process)
        finally:
            usdu_processing.tensor_to_pil = original_tensor_to_pil
        if len(sampled) != 1:
            raise RuntimeError(f"expected one sampled tile, got {len(sampled)}")
        initial_size = (crop_region[2] - crop_region[0], crop_region[3] - crop_region[1])
        tile = sampled[0]
        if tile.size != initial_size:
            tile = tile.resize(initial_size, Image.Resampling.LANCZOS)
        if process.mask_blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(process.mask_blur))
        active_jobs.append(
            {
                "after": _record(np.asarray(result.images[0], dtype=np.uint8)),
                "before": _record(np.asarray(before, dtype=np.uint8)),
                "crop": list(crop_region),
                "mask": _record(np.asarray(mask, dtype=np.uint8)),
                "sample": _record(np.asarray(tile, dtype=np.uint8)),
            }
        )
        return result

    usdu_processing.process_images = capture_process_images
    for name, choices in CASES.items():
        active_jobs = []
        kwargs = {
            key: value
            for key, value in PARAMETERS.items()
            if key not in {"positive_text", "negative_text"}
        }
        kwargs.update(choices)
        output = usdu_nodes.UltimateSDUpscale().upscale(
            image=image,
            model=model,
            positive=positive,
            negative=negative,
            vae=vae,
            upscale_model=upscaler,
            **kwargs,
        )[0]
        pixels = np.rint(output[0].cpu().numpy() * 255.0).astype(np.uint8)
        outputs[name] = _record(pixels)
        jobs[name] = active_jobs
    usdu_processing.process_images = original_process_images

    document = {
        "artifacts": ARTIFACTS,
        "cases": {
            name: {"choices": choices, "output": outputs[name]} for name, choices in CASES.items()
        },
        "environment": {
            "comfyui": COMFYUI_COMMIT,
            "extension": EXTENSION_COMMIT,
            "numpy": np.__version__,
            "pillow": importlib.metadata.version("pillow"),
            "spandrel": importlib.metadata.version("spandrel"),
            "torch": torch.__version__,
            "upstream": UPSTREAM_COMMIT,
        },
        "jobs": jobs,
        "parameters": PARAMETERS,
        "source": _record(source),
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
