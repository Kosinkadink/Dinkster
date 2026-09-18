"""Generate image and mask operation goldens from ComfyUI e20d433a.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies:

    COMFYUI_ROOT=/path/to/ComfyUI \
      PYTHONPATH=/path/to/comfy-dependencies \
      /path/to/python tools/gen_image_mask_goldens.py

The ComfyUI checkout must be clean and pinned to the commit below. Run the
generator twice and compare the printed sha256 before committing the fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

BASELINE = "e20d433a4966dcc88fa5abbae6ace824cb78b263"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "image_mask_e20d433a.json"


def _git(comfy_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(comfy_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tensor_record(tensor: Any) -> dict[str, object]:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "values": [float(value) for value in cpu.reshape(-1).tolist()],
    }


def build_goldens(comfy_root: Path) -> dict[str, object]:
    if _git(comfy_root, "rev-parse", "HEAD") != BASELINE:
        raise RuntimeError(f"ComfyUI must be pinned to {BASELINE}")
    if _git(comfy_root, "status", "--porcelain"):
        raise RuntimeError("ComfyUI checkout must be clean")
    sys.path.insert(0, str(comfy_root))

    import torch  # pyright: ignore[reportMissingImports]
    from comfy_extras import (  # pyright: ignore[reportMissingImports]
        nodes_images,
        nodes_mask,
        nodes_rebatch,
    )
    from nodes import EmptyImage, ImageBatch  # pyright: ignore[reportMissingImports]

    mask = torch.tensor(
        [
            [
                [0.0, 0.2, 0.5, 0.8, 1.0],
                [0.1, 0.3, 0.6, 0.9, 0.4],
                [1.0, 0.7, 0.4, 0.2, 0.0],
                [0.5, 0.0, 1.0, 0.25, 0.75],
            ]
        ],
        dtype=torch.float32,
    )
    source_mask = torch.flip(mask, dims=(1, 2))
    image = torch.tensor(
        [
            [
                [[0.0, 0.25, 0.5, 0.75], [128 / 255, 64 / 255, 1.0, 0.25]],
                [[1.0, 0.5, 0.25, 0.0], [0.501, 0.249, 0.999, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    batch_image = torch.arange(36, dtype=torch.float32).reshape(3, 2, 2, 3) / 35.0
    second_batch = torch.tensor(
        [
            [
                [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
                [[0.3, 0.4, 0.5, 0.6], [0.4, 0.5, 0.6, 0.7]],
            ],
            [
                [[0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.9]],
                [[0.7, 0.8, 0.9, 1.0], [0.8, 0.9, 1.0, 0.1]],
            ],
        ],
        dtype=torch.float32,
    )
    cases: dict[str, object] = {}

    cases["SolidMask"] = _tensor_record(nodes_mask.SolidMask.execute(0.25, 5, 4).result[0])
    cases["InvertMask"] = _tensor_record(nodes_mask.InvertMask.execute(mask).result[0])
    cases["CropMask"] = _tensor_record(nodes_mask.CropMask.execute(mask, 1, 1, 3, 2).result[0])
    cases["FeatherMask"] = _tensor_record(
        nodes_mask.FeatherMask.execute(mask, 2, 1, 2, 1).result[0]
    )
    cases["GrowMask:square"] = _tensor_record(nodes_mask.GrowMask.execute(mask, 2, False).result[0])
    cases["GrowMask:tapered-erode"] = _tensor_record(
        nodes_mask.GrowMask.execute(mask, -1, True).result[0]
    )
    cases["ThresholdMask"] = _tensor_record(nodes_mask.ThresholdMask.execute(mask, 0.5).result[0])
    for operation in ("multiply", "add", "subtract", "and", "or", "xor"):
        cases[f"MaskComposite:{operation}"] = _tensor_record(
            nodes_mask.MaskComposite.execute(mask, source_mask, 1, 1, operation).result[0]
        )
    destination_batch = torch.cat((mask, 1.0 - mask), dim=0)
    cases["MaskComposite:source-singleton"] = _tensor_record(
        nodes_mask.MaskComposite.execute(destination_batch, source_mask, 1, 1, "add").result[0]
    )
    for channel in ("red", "green", "blue", "alpha"):
        cases[f"ImageToMask:{channel}"] = _tensor_record(
            nodes_mask.ImageToMask.execute(image, channel).result[0]
        )
    cases["ImageColorToMask"] = _tensor_record(
        nodes_mask.ImageColorToMask.execute(image, 0x8040FF).result[0]
    )
    cases["MaskToImage"] = _tensor_record(nodes_mask.MaskToImage.execute(mask).result[0])
    cases["EmptyImage"] = _tensor_record(EmptyImage().generate(3, 2, 2, 0x3366CC)[0])
    cases["RepeatImageBatch"] = _tensor_record(
        nodes_images.RepeatImageBatch.execute(batch_image, 2).result[0]
    )
    cases["ImageFromBatch"] = _tensor_record(
        nodes_images.ImageFromBatch.execute(batch_image, -2, 2).result[0]
    )
    cases["ImageBatch"] = _tensor_record(ImageBatch().batch(batch_image[:1], second_batch)[0])
    cases["ImageStitch"] = _tensor_record(
        nodes_images.ImageStitch.execute(
            batch_image[:1],
            "right",
            False,
            2,
            "red",
            second_batch[:, :, :1, :],
        ).result[0]
    )
    rebatched = nodes_rebatch.ImageRebatch.execute([batch_image[:2], batch_image[2:]], [2]).result[
        0
    ]
    cases["RebatchImages"] = {
        "items": [_tensor_record(item) for item in rebatched],
    }

    return {
        "baseline": BASELINE,
        "sourceMask": _tensor_record(mask),
        "sourceImage": _tensor_record(image),
        "sourceBatchImage": _tensor_record(batch_image),
        "secondBatchImage": _tensor_record(second_batch),
        "cases": cases,
    }


def main() -> None:
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    content = (json.dumps(build_goldens(comfy_root.resolve()), indent=2) + "\n").encode()
    OUT.write_bytes(content)
    print(f"{OUT}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
