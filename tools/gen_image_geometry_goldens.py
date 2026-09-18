"""Generate image geometry goldens from ComfyUI e20d433a.

Usage from the Dinkster repository root with a torch interpreter containing
ComfyUI's declared dependencies:

    COMFYUI_ROOT=/path/to/ComfyUI \
      PYTHONPATH=/path/to/comfy-dependencies \
      /path/to/python tools/gen_image_geometry_goldens.py

The ComfyUI checkout must be clean and pinned to the commit below. Run the
generator twice and compare the printed sha256 before committing refreshed
fixtures.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BASELINE = "e20d433a4966dcc88fa5abbae6ace824cb78b263"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "image_geometry_e20d433a.json"


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

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]
    from comfy_extras import (  # pyright: ignore[reportMissingImports]
        nodes_compositing,
        nodes_dataset,
        nodes_images,
        nodes_mask,
        nodes_morphology,
        nodes_post_processing,
    )
    from nodes import (  # pyright: ignore[reportMissingImports]
        ImageInvert,
        ImagePadForOutpaint,
        ImageScale,
        ImageScaleBy,
    )

    source = torch.arange(36, dtype=torch.float32).reshape(1, 3, 4, 3) / 35
    cases: dict[str, object] = {}

    cases["PrimitiveBoundingBox"] = nodes_images.BoundingBox.execute(1, 2, 3, 4).result[0]
    cases["ImageCrop"] = _tensor_record(
        nodes_images.ImageCrop.execute(source, 2, 2, 1, 1).result[0]
    )
    cases["ImageCrop:clamped-origin"] = _tensor_record(
        nodes_images.ImageCrop.execute(source, 2, 2, 20, 20).result[0]
    )
    cases["ImageCropV2"] = _tensor_record(
        nodes_images.ImageCropV2.execute(source, {"x": 1, "y": 0, "width": 3, "height": 2}).result[
            0
        ]
    )
    for rotation in ("none", "90 degrees", "180 degrees", "270 degrees"):
        cases[f"ImageRotate:{rotation}"] = _tensor_record(
            nodes_images.ImageRotate.execute(source, rotation).result[0]
        )
    for flip in ("x-axis: vertically", "y-axis: horizontally"):
        cases[f"ImageFlip:{flip}"] = _tensor_record(
            nodes_images.ImageFlip.execute(source, flip).result[0]
        )

    padded, mask = ImagePadForOutpaint().expand_image(source, 1, 1, 2, 1, 1)
    cases["ImagePadForOutpaint:image"] = _tensor_record(padded)
    cases["ImagePadForOutpaint:mask"] = _tensor_record(mask)
    nodes_images.GetImageSize.hidden = SimpleNamespace(unique_id=None)
    cases["GetImageSize"] = list(nodes_images.GetImageSize.execute(source).result)

    composite_source = torch.flip(source[:, :2, :3, :], dims=(1, 2))
    composite_mask = torch.tensor(
        [[[0.0, 0.25, 0.5], [0.75, 1.0, 0.25]]],
        dtype=torch.float32,
    )
    cases["ImageCompositeMasked"] = _tensor_record(
        nodes_mask.ImageCompositeMasked.execute(
            source,
            composite_source,
            1,
            1,
            False,
            composite_mask,
        ).result[0]
    )
    cases["ImageCompositeMasked:resize-source"] = _tensor_record(
        nodes_mask.ImageCompositeMasked.execute(
            source,
            composite_source,
            0,
            0,
            True,
            composite_mask,
        ).result[0]
    )
    for blend_mode in ("normal", "multiply", "screen", "overlay", "soft_light", "difference"):
        cases[f"ImageBlend:{blend_mode}"] = _tensor_record(
            nodes_post_processing.Blend.execute(
                source,
                composite_source,
                0.4,
                blend_mode,
            ).result[0]
        )

    source_alpha = torch.tensor(
        [[[0.0, 0.25, 0.5], [0.75, 1.0, 0.25]]],
        dtype=torch.float32,
    )
    destination_alpha = torch.linspace(0.0, 1.0, 12, dtype=torch.float32).reshape(1, 3, 4)
    porter_image, porter_mask = nodes_compositing.PorterDuffImageComposite.execute(
        composite_source,
        source_alpha,
        source,
        destination_alpha,
        "SRC_OVER",
    ).result
    cases["PorterDuffImageComposite:image"] = _tensor_record(porter_image)
    cases["PorterDuffImageComposite:mask"] = _tensor_record(porter_mask)
    aspect_source = ((torch.sin(torch.arange(63, dtype=torch.float32) * 0.71) + 1.0) / 2.0).reshape(
        1, 3, 7, 3
    )
    aspect_destination = (
        (torch.cos(torch.arange(60, dtype=torch.float32) * 0.43) + 1.0) / 2.0
    ).reshape(1, 5, 4, 3)
    aspect_source_alpha = torch.linspace(0.0, 1.0, 12, dtype=torch.float32).reshape(1, 6, 2)
    aspect_destination_alpha = torch.linspace(1.0, 0.0, 16, dtype=torch.float32).reshape(1, 2, 8)
    cases["PorterDuffImageComposite:aspect-source"] = _tensor_record(aspect_source)
    cases["PorterDuffImageComposite:aspect-destination"] = _tensor_record(aspect_destination)
    cases["PorterDuffImageComposite:aspect-source-mask"] = _tensor_record(aspect_source_alpha)
    cases["PorterDuffImageComposite:aspect-destination-mask"] = _tensor_record(
        aspect_destination_alpha
    )
    porter_image, porter_mask = nodes_compositing.PorterDuffImageComposite.execute(
        aspect_source,
        aspect_source_alpha,
        aspect_destination,
        aspect_destination_alpha,
        "SRC_OVER",
    ).result
    cases["PorterDuffImageComposite:aspect-image"] = _tensor_record(porter_image)
    cases["PorterDuffImageComposite:aspect-mask"] = _tensor_record(porter_mask)

    cases["ImageInvert"] = _tensor_record(ImageInvert().invert(source)[0])
    cases["NormalizeImages"] = _tensor_record(
        nodes_dataset.NormalizeImagesNode._process(source, 0.4, 0.25)
    )
    cases["AdjustBrightness"] = _tensor_record(
        nodes_dataset.AdjustBrightnessNode._process(source, 1.25)
    )
    cases["AdjustContrast"] = _tensor_record(
        nodes_dataset.AdjustContrastNode._process(source, 1.25)
    )
    cases["ImageBlur"] = _tensor_record(
        nodes_post_processing.Blur.execute(source, 1, 1.0).result[0]
    )
    cases["ImageSharpen"] = _tensor_record(
        nodes_post_processing.Sharpen.execute(source, 1, 1.0, 0.1).result[0]
    )
    filter_stress_source = (
        (torch.sin(torch.arange(1024, dtype=torch.float32) * 0.43) + 1.0) / 2.0
    ).reshape(1, 32, 32, 1)
    cases["ImageFilter:stress-source"] = _tensor_record(filter_stress_source)
    cases["ImageBlur:stress"] = _tensor_record(
        nodes_post_processing.Blur.execute(filter_stress_source, 31, 0.1).result[0]
    )
    cases["ImageSharpen:stress"] = _tensor_record(
        nodes_post_processing.Sharpen.execute(filter_stress_source, 31, 0.1, 5.0).result[0]
    )
    for dither in ("none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"):
        cases[f"ImageQuantize:{dither}"] = _tensor_record(
            nodes_post_processing.Quantize.execute(source, 4, dither).result[0]
        )
    morphology_source = (
        (torch.sin(torch.arange(60, dtype=torch.float32) * 0.73) + 1.0) / 2.0
    ).reshape(1, 4, 5, 3)
    cases["Morphology:source"] = _tensor_record(morphology_source)
    for operation in (
        "erode",
        "dilate",
        "open",
        "close",
        "gradient",
        "bottom_hat",
        "top_hat",
    ):
        cases[f"Morphology:{operation}"] = _tensor_record(
            nodes_morphology.Morphology.execute(morphology_source, operation, 4).result[0]
        )

    rgba = torch.cat(
        (source, torch.linspace(0.0, 1.0, 12, dtype=torch.float32).reshape(1, 3, 4, 1)),
        dim=3,
    )
    split_image, split_mask = nodes_compositing.SplitImageWithAlpha.execute(rgba).result
    cases["SplitImageWithAlpha:image"] = _tensor_record(split_image)
    cases["SplitImageWithAlpha:mask"] = _tensor_record(split_mask)
    cases["JoinImageWithAlpha"] = _tensor_record(
        nodes_compositing.JoinImageWithAlpha.execute(source, destination_alpha).result[0]
    )
    luminance, blue_difference, red_difference = nodes_morphology.ImageRGBToYUV.execute(
        source
    ).result
    cases["ImageRGBToYUV:Y"] = _tensor_record(luminance)
    cases["ImageRGBToYUV:U"] = _tensor_record(blue_difference)
    cases["ImageRGBToYUV:V"] = _tensor_record(red_difference)
    cases["ImageYUVToRGB"] = _tensor_record(
        nodes_morphology.ImageYUVToRGB.execute(
            luminance,
            blue_difference,
            red_difference,
        ).result[0]
    )

    cases["ImageScale:nearest-exact"] = _tensor_record(
        ImageScale().upscale(source, "nearest-exact", 6, 5, "disabled")[0]
    )
    cases["ImageScale:bilinear-center"] = _tensor_record(
        ImageScale().upscale(source, "bilinear", 2, 4, "center")[0]
    )
    cases["ImageScale:bilinear-downscale"] = _tensor_record(
        ImageScale().upscale(source, "bilinear", 3, 2, "disabled")[0]
    )
    cases["ImageScale:area"] = _tensor_record(
        ImageScale().upscale(source, "area", 3, 2, "disabled")[0]
    )
    bicubic_source = (
        (torch.sin(torch.arange(75, dtype=torch.float32) * 1.0452261306532664) + 1.0) / 2.0
    ).reshape(1, 5, 5, 3)
    cases["ImageScale:bicubic-nonlinear-source"] = _tensor_record(bicubic_source)
    cases["ImageScale:bicubic-nonlinear"] = _tensor_record(
        ImageScale().upscale(bicubic_source, "bicubic", 7, 7, "disabled")[0]
    )
    cases["ImageScaleBy:bicubic-nonlinear"] = _tensor_record(
        ImageScaleBy().upscale(bicubic_source, "bicubic", 1.4)[0]
    )
    cases["ImageScaleToTotalPixels:bicubic-nonlinear"] = _tensor_record(
        nodes_post_processing.ImageScaleToTotalPixels.execute(
            bicubic_source,
            "bicubic",
            49 / (1024 * 1024),
            1,
        ).result[0]
    )
    cases["ImageScaleToMaxDimension:bicubic-nonlinear"] = _tensor_record(
        nodes_images.ImageScaleToMaxDimension.execute(bicubic_source, "bicubic", 7).result[0]
    )
    odd_source = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6, 1) / 17
    cases["ImageScale:nearest-exact-center-odd"] = _tensor_record(
        ImageScale().upscale(odd_source, "nearest-exact", 3, 3, "center")[0]
    )
    cases["ImageScale:lanczos"] = _tensor_record(
        ImageScale().upscale(source, "lanczos", 5, 4, "disabled")[0]
    )
    cases["ImageScaleBy"] = _tensor_record(ImageScaleBy().upscale(source, "bicubic", 1.5)[0])
    cases["ImageScaleToTotalPixels"] = _tensor_record(
        nodes_post_processing.ImageScaleToTotalPixels.execute(
            source,
            "nearest-exact",
            48 / (1024 * 1024),
            2,
        ).result[0]
    )
    cases["ImageScaleToMaxDimension"] = _tensor_record(
        nodes_images.ImageScaleToMaxDimension.execute(source, "nearest-exact", 8).result[0]
    )

    return {
        "baseline": BASELINE,
        "versions": {
            "numpy": np.__version__,
            "pillow": importlib.metadata.version("Pillow"),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "source": _tensor_record(source),
        "cases": cases,
    }


def main() -> None:
    configured = os.environ.get("COMFYUI_ROOT")
    comfy_root = Path(configured) if configured else REPO.parent / "ComfyUI"
    content = (json.dumps(build_goldens(comfy_root.resolve()), indent=2) + "\n").encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(content)
    print(f"{OUT}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
