"""Generate CPU layer goldens with COMFYUI_ROOT pinned to the reference commit.

Run with .venv-torch/bin/python tools/gen_layer_document_goldens.py.
Only the upstream compositor functions are loaded; no server or node registration runs.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

REFERENCE_COMMIT = "f00bfd610cb001381603669e2cc01160ae37aaf3"


def main() -> None:
    root = Path(os.environ["COMFYUI_ROOT"]).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != REFERENCE_COMMIT or subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"]
    ):
        raise RuntimeError("ComfyUI reference checkout must be clean and pinned")
    sys.path.insert(0, str(root))
    torch = importlib.import_module("torch")
    blend = importlib.import_module("comfy_extras.compositor_blend")
    colors = importlib.import_module("comfy_extras.color_util")
    source = root / "comfy_extras/nodes_compositor.py"
    tree = ast.parse(source.read_text())
    functions = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            or isinstance(node, ast.ClassDef)
            and node.name in ("AddLayer", "LayersFromBoundingBoxes", "ImageCompositor")
        ],
        type_ignores=[],
    )
    namespace: dict[str, Any] = dict(
        np=np,
        torch=torch,
        Image=Image,
        hashlib=hashlib,
        json=json,
        math=math,
        MAX_RESOLUTION=16384,
        MAX_LAYERS=50,
        OPAQUE_EPSILON=1e-3,
        _HEX_DIGITS=set("0123456789abcdef"),
        hex_to_rgb=colors.hex_to_rgb,
        io=SimpleNamespace(ComfyNode=object, NodeOutput=lambda *values, **_kwargs: values),
        UI=SimpleNamespace(
            PreviewImage=lambda *_args, **_kwargs: SimpleNamespace(values=[], as_dict=lambda: {})
        ),
    )
    namespace.update(
        {
            name: getattr(blend, name)
            for name in (
                "_LAYER_MODES",
                "blend_composite",
                "linear_to_srgb",
                "placed_bounds",
                "resolve_mode",
                "srgb_to_linear",
            )
        }
    )
    boxes = ast.parse((root / "comfy_extras/nodes_bounding_boxes.py").read_text())
    exec(
        compile(
            ast.Module(
                body=[node for node in boxes.body if isinstance(node, ast.FunctionDef)],
                type_ignores=[],
            ),
            "nodes_bounding_boxes.py",
            "exec",
        ),
        namespace,
    )
    exec(compile(functions, str(source), "exec"), namespace)
    cases = []
    for opacity in (1.0, 0.5):
        layers = [
            {
                "image": np.full((1, 3, 4, 3), (32, 64, 128), dtype=np.float32) / 255,
                "type": "raster",
                "z_index": 0,
                "name": "Background",
            },
            {
                "image": np.full((1, 2, 2, 3), (220, 100, 50), dtype=np.float32) / 255,
                "mask": np.array([[[0, 1], [0, 0]]], dtype=np.float32),
                "type": "raster",
                "x": 1,
                "y": 1,
                "z_index": 10,
                "name": "Subject",
                "opacity": opacity,
            },
        ]
        record = {
            "version": 1,
            "canvas": [4, 3],
            "layers": [
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in layer.items()
                }
                for layer in layers
            ],
        }
        tensors = [
            {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in layer.items()
            }
            for layer in layers
        ]
        frames = namespace["expand_item_frames"](namespace["document_items"]({"layers": tensors}))
        alphas = [namespace["frame_alpha"](frame["tensor"], frame["mask"]) for frame in frames]
        state = namespace["state_from_items"](frames, (4, 3))
        rgba = namespace["composite_from_state"](
            [frame["tensor"] for frame in frames], state, alphas
        )
        image, mask = namespace["composite_outputs"](rgba)
        cases.append(
            {
                "layers": record,
                "image": image.tolist(),
                "mask": mask.tolist(),
                "inputs": namespace["input_fingerprints"](frames, alphas),
            }
        )
    alias_cases = []
    for name in ("AddLayer", "LayersFromBoundingBoxes", "ImageCompositor"):
        background = {
            "version": 1,
            "canvas": [4, 3],
            "layers": [{**cases[0]["layers"]["layers"][0], "z_index": 10}],
        }
        inputs: dict[str, Any] = {"layers": background}
        if name == "ImageCompositor":
            inputs["layers"] = cases[1]["layers"]
            inputs["compositor"] = {}
        else:
            inputs.update(
                image=cases[0]["layers"]["layers"][1]["image"],
                mask=cases[0]["layers"]["layers"][1]["mask"],
            )
            if name == "AddLayer":
                inputs.update(x=1, y=1, z_index=11, opacity=0.5, name="Subject", flip_h=True)
            else:
                inputs["bboxes"] = json.dumps([{"x": 1, "y": 1, "width": 2, "height": 2}])

        def tensors_from_record(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: torch.tensor(item, dtype=torch.float32)
                    if key in ("image", "mask")
                    else tensors_from_record(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [tensors_from_record(item) for item in value]
            return value

        result = namespace[name].execute(**tensors_from_record(inputs))
        image, mask = (
            result if name == "ImageCompositor" else namespace["ImageCompositor"].execute(result[0])
        )
        alias_cases.append(
            {"node": name, "inputs": inputs, "image": image.tolist(), "mask": mask.tolist()}
        )
    output = Path(__file__).resolve().parents[1] / "tests/fixtures/layer_document_goldens.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(
            {"reference": REFERENCE_COMMIT, "cases": cases, "aliasCases": alias_cases},
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    print(hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
