"""Compare new RenderSplat cells against the immutable issue 511 reference on CPU.

Run with the CPU pack-test interpreter and source paths on PYTHONPATH. The
receipt is append-only: cells already recorded there are never executed again.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.splat_render import RenderSplat

REFERENCE = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
SOURCE_SHA256 = "151e1cd2bf5d25c9a5d9b6a9addc1afb8f7a0895cbc7b9ee1f69a869aa1e3e4b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == SOURCE_SHA256
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    utils = importlib.import_module("dinkster_inference.utils")
    assert management.get_torch_device().type == "cpu"
    names = {
        "_srgb_to_linear",
        "_linear_to_srgb",
        "_real_len",
        "_hex_to_rgb",
        "_quantile",
        "_gaussian_item",
        "_quat_to_mat",
        "_quat_mul",
        "_mat_to_quat",
        "_view_matrix_t",
        "_camera_basis",
        "_lookat_quat_wxyz",
        "_lookat_camera_info",
        "_orbit_camera_info",
        "_orbit_camera_info_yaw",
        "_gauss_blur",
        "_render_gaussian",
        "RenderSplat",
    }
    namespace = {
        "torch": torch,
        "math": math,
        "_C0": 0.28209479177387814,
        "comfy": SimpleNamespace(model_management=management, utils=utils),
        "Types": SimpleNamespace(SPLAT=SimpleNamespace),
        "IO": SimpleNamespace(ComfyNode=object, NodeOutput=lambda *values: values),
    }
    definitions = [
        node
        for node in ast.parse(args.source.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(definitions) == len(names)
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(args.source), "exec"), namespace
    )
    receipt = (
        json.loads(args.receipt.read_text())
        if args.receipt.exists()
        else {
            "reference": REFERENCE,
            "source_sha256": SOURCE_SHA256,
            "cells": {},
        }
    )
    assert receipt["reference"] == REFERENCE and receipt["source_sha256"] == SOURCE_SHA256
    for projection in ("perspective", "orthographic", "default"):
        for style in ("color", "clay", "depth", "normal"):
            key = f"{projection}-{style}"
            if key in receipt["cells"]:
                continue
            value = SimpleNamespace(
                positions=torch.tensor([[[0.1, -0.05, 0.0], [-0.08, 0.07, 0.4], [0, 0, 0]]]),
                scales=torch.tensor([[[0.12, 0.06, 0.025], [0.06, 0.1, 0.04], [1, 1, 1]]]),
                rotations=torch.tensor([[[1, 0, 0, 0], [0.96, 0.1, 0.2, -0.15], [1, 0, 0, 0]]]),
                opacities=torch.tensor([[[0.7], [0.8], [1.0]]]),
                sh=torch.tensor([[[[0.4, -0.3, 0.1]], [[-0.2, 0.5, 0.0]], [[0, 0, 0]]]]),
                counts=torch.tensor([2]),
            )
            camera = (
                None
                if projection == "default"
                else {
                    "position": {"x": 0.0, "y": 0.0, "z": 3.0},
                    "target": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    "cameraType": projection,
                    "fov": 35.0,
                    "zoom": 1.1,
                }
            )
            parameters = dict(
                width=80,
                height=64,
                frames=-3,
                splat_scale=1.1,
                sharpen=2.0,
                headlight_shading=0.7,
                opacity_threshold=0.0,
                render_style=style,
                background="#204080",
                camera_info=camera,
            )
            expected = namespace["RenderSplat"].execute(value, **parameters)
            actual = RenderSplat.execute(splat=value, **parameters)
            hashes = {}
            for field, reference in zip(("IMAGE", "MASK"), expected, strict=True):
                tensor = actual[field]
                torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
                hashes[field] = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
            receipt["cells"][key] = {"bit_exact": True, "sha256": hashes}
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            print(f"Accepted new RenderSplat cell: {key}", flush=True)


if __name__ == "__main__":
    main()
