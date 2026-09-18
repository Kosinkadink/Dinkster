"""Generate image alias replay vectors with .venv-torch and COMFYUI_ROOT.

The source checkout must be clean at BASELINE. Run twice and compare SHA256.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
OUT = Path(__file__).resolve().parents[1] / "tests/goldens/image_aliases_b78cec87.json"


def main() -> None:
    root = Path(os.environ["COMFYUI_ROOT"]).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if head != BASELINE or dirty:
        raise RuntimeError(f"ComfyUI must be clean and pinned to {BASELINE}")
    sys.path.insert(0, str(root))
    from comfy.cli_args import args

    args.cpu = True
    import torch
    from comfy_extras.nodes_images import ResizeAndPadImage
    from comfy_extras.nodes_post_processing import BatchImagesNode

    def record(tensor: Any) -> dict[str, object]:
        return {"shape": list(tensor.shape), "values": tensor.cpu().flatten().tolist()}

    rgb = torch.arange(45, dtype=torch.float32).reshape(1, 3, 5, 3) / 44
    rgba = torch.arange(240, dtype=torch.float32).reshape(1, 6, 10, 4) / 239
    resize = []
    for method in ("area", "bicubic", "nearest-exact", "bilinear", "lanczos"):
        for color in ("black", "white"):
            result = ResizeAndPadImage.execute(rgb, 4, 4, color, method)[0]
            resize.append({"method": method, "color": color, "image": record(result)})
    document = {
        "baseline": BASELINE,
        "torch": torch.__version__,
        "rgb": record(rgb),
        "rgba": record(rgba),
        "batch": record(BatchImagesNode.execute({"image1": rgb, "image2": rgba})[0]),
        "resize": resize,
    }
    data = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    OUT.write_bytes(data)
    print(f"{OUT}: sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
