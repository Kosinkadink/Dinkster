"""Generate exact tensor resize vectors from pinned ComfyUI source.

From the repository root, using Python 3.12, torch 2.13, numpy 2.5.1,
and Pillow 12.0.0:

    .venv-torch/bin/python tools/gen_tensor_resize_goldens.py --comfyui PATH

PATH must contain the immutable reference commit below. The source is read
with git show; working files and the checkout's HEAD are not used for goldens.
Only the five tensor helpers are compiled, with their real torch/numpy/PIL
dependencies. No ComfyUI bootstrap, substitute kernels, or GPU is involved.
Run twice and compare the printed SHA-256 before committing the fixture.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REFERENCE = "947c2749dd04c51ef0e21b069544d8b0b4f9b411"
FUNCTIONS = {"common_upscale", "bislerp", "lanczos", "resize_to_batch_size", "repeat_to_batch_size"}
OUT = (
    Path(__file__).resolve().parents[1]
    / "packages/dinkster-inference-torch/tests/goldens/tensor_resize_947c2749.json"
)


def reference_functions(root: Path) -> tuple[dict[str, Any], str]:
    source = subprocess.run(
        ["git", "-C", str(root), "show", f"{REFERENCE}:comfy/utils.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tree = ast.parse(source)
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    if {node.name for node in definitions} != FUNCTIONS:
        raise RuntimeError("reference tensor helpers are incomplete")
    namespace: dict[str, Any] = {"torch": torch, "np": np, "Image": Image, "math": math}
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), "comfy/utils.py", "exec"), namespace
    )
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def record(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "values": tensor.flatten().tolist(),
    }


def generate(root: Path) -> dict[str, object]:
    functions, source_sha = reference_functions(root)
    cases: dict[str, object] = {}

    def add(name: str, function: str, tensor: torch.Tensor, **kwargs: object) -> None:
        cases[name] = {
            "function": function,
            "input": record(tensor),
            "kwargs": kwargs,
            "output": record(functions[function](tensor.clone(), **kwargs)),
        }

    for dtype in (torch.float32, torch.float16):
        source = (torch.arange(90).reshape(2, 3, 3, 5) - 30).to(dtype) / 32
        for method in ("nearest-exact", "bilinear", "bicubic", "area", "bislerp", "lanczos"):
            for crop in ("center", "disabled"):
                add(
                    f"{dtype}:{method}:{crop}",
                    "common_upscale",
                    source,
                    width=4,
                    height=6,
                    upscale_method=method,
                    crop=crop,
                )
        for channels in (1, 4):
            source = torch.arange(channels * 15).reshape(1, channels, 3, 5).to(dtype) / 16
            add(f"lanczos:{dtype}:channels{channels}", "lanczos", source, width=7, height=2)
        source = torch.arange(120).reshape(1, 3, 2, 4, 5).to(dtype) / 64
        add(
            f"video:{dtype}",
            "common_upscale",
            source,
            width=3,
            height=6,
            upscale_method="bilinear",
            crop="center",
        )
        source = torch.zeros((1, 4, 3, 3), dtype=dtype)
        add(f"bislerp:zero:{dtype}", "bislerp", source, width=5, height=2)
    source = torch.arange(24).reshape(4, 2, 3)
    for function in ("resize_to_batch_size", "repeat_to_batch_size"):
        for batch in (0, 1, 2, 3, 4, 7):
            add(f"{function}:{batch}", function, source, batch_size=batch)
    for dim in (1, 2):
        add(f"repeat:axis{dim}", "repeat_to_batch_size", source, batch_size=5, dim=dim)
    return {
        "reference": {"commit": REFERENCE, "source_sha256": source_sha},
        "runtime": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    payload = (
        json.dumps(generate(args.comfyui), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    args.output.write_bytes(payload)
    print(f"{hashlib.sha256(payload).hexdigest()}  {args.output}")


if __name__ == "__main__":
    main()
