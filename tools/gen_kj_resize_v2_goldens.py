"""Generate ImageResizeKJv2 goldens from pinned ComfyUI and KJNodes sources.

Usage from the Dinkster repository root with a torch >= 2.10 interpreter:

    COMFYUI_ROOT=/path/to/ComfyUI \
      KJ_ROOT=/path/to/ComfyUI-KJNodes \
      /path/to/python tools/gen_kj_resize_v2_goldens.py

Both checkouts must be clean and pinned to the commits below. Run the
generator twice and compare the printed sha256 before committing the fixture.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

COMFY_BASELINE = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
KJ_BASELINE = "827fe6ee0ed7348d8daa988ed852bedf1272380c"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "kj_resize_v2_827fe6ee.json"


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_checkout(root: Path, revision: str, name: str) -> None:
    if _git(root, "rev-parse", "HEAD") != revision:
        raise RuntimeError(f"{name} must be pinned to {revision}")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError(f"{name} checkout must be clean")


def _module(name: str, **attributes: object) -> ModuleType:
    result = ModuleType(name)
    result.__dict__.update(attributes)
    sys.modules[name] = result
    return result


def _package(name: str, path: Path | None = None, **attributes: object) -> ModuleType:
    result = _module(name, **attributes)
    result.__path__ = [] if path is None else [str(path)]  # type: ignore[attr-defined]
    return result


def _reference_common_upscale(comfy_root: Path, torch: Any, np: Any, image_type: Any) -> Any:
    source_path = comfy_root / "comfy" / "utils.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    selected: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"lanczos", "common_upscale"}
    ]
    if {node.name for node in selected if isinstance(node, ast.FunctionDef)} != {
        "lanczos",
        "common_upscale",
    }:
        raise RuntimeError("pinned ComfyUI resize functions were not found")
    namespace: dict[str, Any] = {"torch": torch, "np": np, "Image": image_type}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["common_upscale"]


def _load_kj_resize(comfy_root: Path, kj_root: Path, torch: Any, np: Any) -> Any:
    from PIL import Image

    common_upscale = _reference_common_upscale(comfy_root, torch, np, Image)
    _package("comfy")
    _module("comfy.cli_args", args=SimpleNamespace())
    _module(
        "comfy.utils",
        common_upscale=common_upscale,
        ProgressBar=object,
        tiled_scale_multidim=lambda *args, **kwargs: None,
    )
    _module("comfy.model_management", get_torch_device=lambda: torch.device("cpu"))
    _package("comfy_extras")
    _module("comfy_extras.nodes_mask", composite=lambda *args, **kwargs: None)
    _module("nodes", MAX_RESOLUTION=16_384, SaveImage=object)
    _module("node_helpers")
    _module("folder_paths")
    _module("server", PromptServer=None, BinaryEventTypes=None)
    _package("comfy_api")
    io = SimpleNamespace(
        **{
            name: object
            for name in (
                "Audio",
                "Boolean",
                "Combo",
                "ComfyNode",
                "DynamicCombo",
                "Float",
                "FolderType",
                "Hidden",
                "Image",
                "Int",
                "Latent",
                "Mask",
                "MatchType",
                "MultiType",
                "NodeOutput",
                "Schema",
                "String",
                "Vae",
                "Video",
            )
        }
    )
    _module(
        "comfy_api.latest",
        io=io,
        InputImpl=SimpleNamespace(),
        Types=SimpleNamespace(),
        ui=SimpleNamespace(),
    )

    package = "kj_resize_reference"
    _package(package, kj_root)
    _package(f"{package}.nodes", kj_root / "nodes")
    _package(f"{package}.utility", kj_root / "utility")
    module_name = f"{package}.nodes.image_nodes"
    spec = importlib.util.spec_from_file_location(module_name, kj_root / "nodes" / "image_nodes.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned KJNodes image module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.ImageResizeKJv2


def _tensor_record(tensor: Any) -> dict[str, object]:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "values": [float(value) for value in cpu.reshape(-1).tolist()],
    }


def build_goldens(comfy_root: Path, kj_root: Path) -> dict[str, object]:
    _require_checkout(comfy_root, COMFY_BASELINE, "ComfyUI")
    _require_checkout(kj_root, KJ_BASELINE, "ComfyUI-KJNodes")

    import numpy as np
    import torch  # pyright: ignore[reportMissingImports]

    resize_type = _load_kj_resize(comfy_root, kj_root, torch, np)
    landscape = ((torch.sin(torch.arange(105, dtype=torch.float32) * 0.37) + 1.0) / 2.0).reshape(
        1, 5, 7, 3
    )
    portrait = ((torch.cos(torch.arange(84, dtype=torch.float32) * 0.41) + 1.0) / 2.0).reshape(
        1, 7, 4, 3
    )
    landscape_mask = torch.linspace(0.0, 1.0, 35, dtype=torch.float32).reshape(1, 5, 7)
    portrait_mask = torch.linspace(1.0, 0.0, 28, dtype=torch.float32).reshape(1, 7, 4)
    sources = {"landscape": landscape, "portrait": portrait}
    masks = {"landscape": landscape_mask, "portrait": portrait_mask}
    specifications: dict[str, dict[str, object]] = {
        "stretch_landscape": {
            "source": "landscape",
            "mask": "landscape",
            "width": 10,
            "height": 8,
            "mode": "stretch",
            "interpolation": "nearest-exact",
            "divisible_by": 2,
            "pad_color": "0, 0, 0",
            "anchor": "center",
        },
        "resize_portrait": {
            "source": "portrait",
            "mask": "portrait",
            "width": 10,
            "height": 8,
            "mode": "resize",
            "interpolation": "nearest-exact",
            "divisible_by": 1,
            "pad_color": "0, 0, 0",
            "anchor": "center",
        },
        "crop_landscape": {
            "source": "landscape",
            "mask": "landscape",
            "width": 8,
            "height": 8,
            "mode": "crop",
            "interpolation": "nearest-exact",
            "divisible_by": 1,
            "pad_color": "0, 0, 0",
            "anchor": "right",
        },
        "pad_rgb_landscape": {
            "source": "landscape",
            "mask": None,
            "width": 11,
            "height": 11,
            "mode": "pad",
            "interpolation": "nearest-exact",
            "divisible_by": 2,
            "pad_color": "17, 91, 203",
            "anchor": "top",
        },
        "pad_rgb_portrait": {
            "source": "portrait",
            "mask": "portrait",
            "width": 9,
            "height": 9,
            "mode": "pad",
            "interpolation": "nearest-exact",
            "divisible_by": 1,
            "pad_color": "0.5, 0.25, 1.0",
            "anchor": "right",
        },
        "pad_edge_landscape": {
            "source": "landscape",
            "mask": "landscape",
            "width": 11,
            "height": 9,
            "mode": "pad_edge",
            "interpolation": "nearest-exact",
            "divisible_by": 1,
            "pad_color": "0, 0, 0",
            "anchor": "bottom",
        },
        "pad_edge_bicubic_landscape": {
            "source": "landscape",
            "mask": "landscape",
            "width": 11,
            "height": 9,
            "mode": "pad_edge",
            "interpolation": "bicubic",
            "divisible_by": 1,
            "pad_color": "0, 0, 0",
            "anchor": "bottom",
        },
        "pad_edge_pixel_portrait": {
            "source": "portrait",
            "mask": None,
            "width": 9,
            "height": 10,
            "mode": "pad_edge_pixel",
            "interpolation": "nearest-exact",
            "divisible_by": 2,
            "pad_color": "0, 0, 0",
            "anchor": "left",
        },
        "pillarbox_blur_landscape": {
            "source": "landscape",
            "mask": "landscape",
            "width": 11,
            "height": 11,
            "mode": "pillarbox_blur",
            "interpolation": "bilinear",
            "divisible_by": 1,
            "pad_color": "0, 0, 0",
            "anchor": "center",
        },
        "total_pixels_landscape": {
            "source": "landscape",
            "mask": "landscape",
            "width": 19,
            "height": 17,
            "mode": "total_pixels",
            "interpolation": "nearest-exact",
            "divisible_by": 2,
            "pad_color": "0, 0, 0",
            "anchor": "center",
        },
        "total_pixels_portrait": {
            "source": "portrait",
            "mask": "portrait",
            "width": 13,
            "height": 19,
            "mode": "total_pixels",
            "interpolation": "nearest-exact",
            "divisible_by": 3,
            "pad_color": "0, 0, 0",
            "anchor": "center",
        },
    }

    cases: dict[str, object] = {}
    for name, specification in specifications.items():
        source_name = cast("str", specification["source"])
        mask_name = cast("str | None", specification["mask"])
        arguments = {
            key: value
            for key, value in specification.items()
            if key not in ("source", "mask", "anchor")
        }
        result = resize_type().resize(
            sources[source_name],
            arguments["width"],
            arguments["height"],
            arguments["mode"],
            arguments["interpolation"],
            arguments["divisible_by"],
            arguments["pad_color"],
            specification["anchor"],
            None,
            device="cpu",
            mask=None if mask_name is None else masks[mask_name],
        )
        cases[name] = {
            **specification,
            "image": _tensor_record(result[0]),
            "output_width": result[1],
            "output_height": result[2],
            "output_mask": _tensor_record(result[3]),
        }

    return {
        "baselines": {"comfyui": COMFY_BASELINE, "comfyui-kjnodes": KJ_BASELINE},
        "versions": {
            "numpy": np.__version__,
            "pillow": importlib.metadata.version("Pillow"),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "sources": {name: _tensor_record(source) for name, source in sources.items()},
        "masks": {name: _tensor_record(mask) for name, mask in masks.items()},
        "cases": cases,
    }


def main() -> None:
    configured_comfy = os.environ.get("COMFYUI_ROOT")
    configured_kj = os.environ.get("KJ_ROOT")
    if configured_comfy is None or configured_kj is None:
        raise RuntimeError("COMFYUI_ROOT and KJ_ROOT are required")
    content = (
        json.dumps(
            build_goldens(Path(configured_comfy).resolve(), Path(configured_kj).resolve()),
            indent=2,
        )
        + "\n"
    ).encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(content)
    print(f"{OUT}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
