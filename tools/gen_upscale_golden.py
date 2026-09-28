"""Generate upscale-model parity vectors with the pinned ComfyUI implementation.

Usage from the Dinkster repository root::

    /path/to/python tools/gen_upscale_golden.py \
        /path/to/ComfyUI-at-a1079ba1 \
        /path/to/RealESRGAN_x4plus_anime_6B.pth \
        /path/to/realesr-general-x4v3.pth

Use Python 3.12 with the package versions recorded in the generated payload.
The reference import requires spandrel plus ComfyUI's comfy.utils import
closure (safetensors, Pillow, tqdm, einops, comfy-aimdo, comfy-kitchen).

Model provenance (both verified against these SHA-256 pins):

- RealESRGAN_x4plus_anime_6B.pth (17,938,799 bytes) from
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
- realesr-general-x4v3.pth (4,885,111 bytes) from
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from golden_platform import cpu_identity

BASELINE = "a1079ba16f2674734b065eb036fbfdddaa321a4d"
MODELS = {
    "anime6b": {
        "sha256": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        "blake3": "blake3:717c1bcb17218786f29dd5377dd53a905fb5ec33c6ca12cb8dc0e3f2f18fa1b7",
    },
    "general": {
        "sha256": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        "blake3": "blake3:c24daac1228a6a1d035f71368d28cd39fcb9457347b1137ef94f9d685bae0e0b",
    },
}
TILINGS = {"full": (512, 32), "tiled": (32, 8)}
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "upscale_realesrgan_a1079ba1.json"


def _source() -> np.ndarray:
    height, width = 40, 56
    rows = np.arange(height, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[..., 0] = rows * (255.0 / (height - 1))
    frame[..., 1] = columns * (255.0 / (width - 1))
    frame[..., 2] = (rows + columns) % 256.0
    frame[6:18, 8:26, 0] = 255.0
    frame[6:18, 8:26, 1] = 32.0
    frame[24:36, 30:50, 2] = 240.0
    inside = (rows - 20.0) ** 2 + (columns - 42.0) ** 2 <= 81.0
    frame[inside] = (16.0, 224.0, 96.0)
    return frame.astype(np.uint8)


def _record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array, dtype=np.uint8)
    return {
        "shape": list(contiguous.shape),
        "uint8Base64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _check_reference(reference: Path) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if head != BASELINE:
        raise SystemExit(f"ComfyUI must be checked out at {BASELINE}, got {head}")
    status = subprocess.check_output(
        ["git", "-C", str(reference), "status", "--porcelain"],
        text=True,
    )
    if status:
        raise SystemExit("ComfyUI checkout must be clean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("anime6b", type=Path)
    parser.add_argument("general", type=Path)
    args = parser.parse_args()
    reference = args.reference.resolve()
    _check_reference(reference)
    model_paths = {"anime6b": args.anime6b, "general": args.general}
    for name, path in model_paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODELS[name]["sha256"]:
            raise SystemExit(f"{name} SHA-256 does not match the pinned Real-ESRGAN artifact")

    sys.path.insert(0, str(reference))
    import comfy.utils
    import spandrel
    from spandrel import ImageModelDescriptor, ModelLoader

    source = _source()
    in_img = torch.from_numpy(source[None].astype(np.float32) / 255.0).movedim(-1, -3)
    cases: dict[str, dict[str, object]] = {}
    scales: dict[str, int] = {}
    for name, path in model_paths.items():
        sd = comfy.utils.load_torch_file(str(path), safe_load=True)
        descriptor = ModelLoader().load_from_state_dict(sd).eval()
        if not isinstance(descriptor, ImageModelDescriptor):
            raise SystemExit(f"{name} must load as a single-image model")
        scales[name] = int(descriptor.scale)
        outputs: dict[str, object] = {}
        for tiling, (tile, overlap) in TILINGS.items():
            upscaled = comfy.utils.tiled_scale(
                in_img,
                lambda a, descriptor=descriptor: descriptor(a.float()),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=descriptor.scale,
                output_device="cpu",
            )
            clamped = torch.clamp(upscaled.movedim(-3, -1), min=0, max=1.0)
            outputs[tiling] = _record(
                np.rint(clamped[0].numpy() * 255.0).astype(np.uint8),
            )
        cases[name] = outputs

    document = {
        "baseline": BASELINE,
        "generationCpu": cpu_identity(),
        "models": MODELS,
        "scales": scales,
        "tilings": {name: list(pair) for name, pair in TILINGS.items()},
        "numpy": np.__version__,
        "spandrel": spandrel.__version__,
        "torch": torch.__version__,
        "source": _record(source),
        "cases": cases,
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256:{hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
