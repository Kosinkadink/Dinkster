"""Generate LoadImage goldens with a torch >= 2.10 ComfyUI dependency environment.

Run COMFYUI_ROOT=/clean/ComfyUI .venv-comfy/bin/python tools/gen_image_load_goldens.py.
The checkout must match BASELINE. Run twice and compare the printed SHA256.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BASELINE = "e20d433a4966dcc88fa5abbae6ace824cb78b263"
OUT = Path(__file__).resolve().parent.parent / "tests/goldens/image_load_e20d433a.json"


def main() -> None:
    root = Path(os.environ["COMFYUI_ROOT"]).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if head != BASELINE or dirty:
        raise RuntimeError(f"ComfyUI must be clean and pinned to {BASELINE}")
    sys.path.insert(0, str(root))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options  # pyright: ignore[reportMissingImports]

    comfy.options.enable_args_parsing()
    import folder_paths  # pyright: ignore[reportMissingImports]
    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from nodes import LoadImage  # pyright: ignore[reportMissingImports]
    from PIL import Image

    def record(tensor: Any) -> dict[str, object]:
        return {"shape": list(tensor.shape), "values": tensor.cpu().flatten().tolist()}

    cases: dict[str, object] = {}
    pixels = np.array([[[10, 20, 30, 0], [40, 50, 60, 64], [70, 80, 90, 255]]], np.uint8)
    with tempfile.TemporaryDirectory() as directory:
        folder_paths.set_input_directory(directory)
        for layout in ("rgb", "rgba"):
            path = Path(directory) / f"{layout}.png"
            Image.fromarray(pixels[..., :3] if layout == "rgb" else pixels).save(path)
            image, mask = LoadImage().load_image(path.name)
            cases[layout] = {
                "png": base64.b64encode(path.read_bytes()).decode("ascii"),
                "image": record(image),
                "mask": record(mask),
            }
        for mode in ("RGB", "RGBA"):
            for format_name in ("PNG", "WEBP", "TIFF", "GIF"):
                path = Path(directory) / f"frames.{format_name.lower()}"
                size = (32, 2) if format_name == "GIF" else (3, 2)
                frames = [Image.new(mode, size, color) for color in ("red", "blue")]
                if mode == "RGBA":
                    frames[0].putalpha(128)
                frames[0].save(
                    path,
                    format=format_name,
                    save_all=True,
                    append_images=frames[1:],
                    duration=100,
                    lossless=True,
                )
                image, mask = LoadImage().load_image(path.name)
                cases[f"batch_{mode.lower()}_{format_name.lower()}"] = {
                    "file": base64.b64encode(path.read_bytes()).decode("ascii"),
                    "image": record(image),
                    "mask": record(mask),
                }
    data = (
        json.dumps(
            {"baseline": BASELINE, "torch": torch.__version__, "cases": cases},
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    OUT.write_text(data, encoding="utf-8")
    print(hashlib.sha256(data.encode()).hexdigest())


if __name__ == "__main__":
    main()
