"""Compare new CPU splat meshing cells with an immutable reference, once per cell."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from dinkster_api.v1 import initialize_inference_process, materialize_mesh
from dinkster_nodes_comfy.splat_mesh import SplatToMesh
from scipy.ndimage import map_coordinates
from scipy.ndimage import maximum as ndi_maximum
from scipy.ndimage import minimum as ndi_minimum
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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
    assert management.get_torch_device().type == "cpu"
    names = {
        "_real_len",
        "_inverse_covariance",
        "_splat_density",
        "_connected_components_gpu",
        "_clean_components_gpu",
        "_clean_components",
        "_surface_nets",
        "_otsu_level",
        "_taubin_smooth",
        "_sample_vertex_colours_gpu",
        "_gaussian_to_mesh",
    }
    namespace = dict(
        torch=torch,
        np=np,
        coo_matrix=coo_matrix,
        connected_components=connected_components,
        _ndi_minimum=ndi_minimum,
        _ndi_maximum=ndi_maximum,
        map_coordinates=map_coordinates,
        comfy=SimpleNamespace(model_management=management),
        Types=SimpleNamespace(SPLAT=object),
        _C0=0.28209479177387814,
    )
    definitions = [
        node
        for node in ast.parse(args.source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(definitions) == len(names)
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(args.source), "exec"), namespace
    )
    receipt = (
        json.loads(args.receipt.read_text())
        if args.receipt.exists()
        else {"reference": REFERENCE, "source_sha256": SOURCE_SHA256, "cells": {}}
    )
    assert receipt["reference"] == REFERENCE and receipt["source_sha256"] == SOURCE_SHA256
    for key, smooth, bias, minimum, cutoff, sharpen in (
        ("anisotropic-rotated", 0, 0.4, 0, 0.02, 2.0),
        ("taubin-unsharpened", 3, 1.0, 0, 0.02, 1.0),
        ("component-clean", 0, 0.4, 8, 0.02, 3.0),
        ("high-level", 0, 2.0, 0, 0.02, 2.0),
        ("opacity-empty", 0, 0.4, 0, 0.99, 2.0),
    ):
        if key in receipt["cells"]:
            continue
        source = SimpleNamespace(
            positions=torch.tensor([[[0.1, 0.2, -0.05], [0.25, -0.1, 0.03], [100, 200, 300]]]),
            scales=torch.tensor([[[0.12, 0.07, 0.03], [0.05, 0.1, 0.08], [1, 1, 1]]]),
            rotations=torch.tensor(
                [[[0.96, 0.1, 0.2, -0.15], [0.9, -0.2, 0.1, 0.3], [1, 0, 0, 0]]]
            ),
            opacities=torch.tensor([[[0.8], [0.6], [1.0]]]),
            sh=torch.tensor([[[[0.4, -0.3, 0.1]], [[-0.2, 0.5, 0]], [[0, 0, 0]]]]),
            counts=torch.tensor([2]),
        )
        expected = namespace["_gaussian_to_mesh"](
            source, 0, 64, 5, smooth, bias, minimum, cutoff, sharpen, torch.device("cpu")
        )
        actual = materialize_mesh(
            SplatToMesh.execute(
                splat=source,
                resolution=64,
                kernel=5,
                smooth=smooth,
                level=bias,
                min_component=minimum,
                min_opacity=cutoff,
                color_sharpen=sharpen,
            )["mesh"]
        )
        if expected is None:
            expected = (
                torch.zeros((0, 3)),
                torch.zeros((0, 3), dtype=torch.int64),
                torch.zeros((0, 3)),
            )
        fields = (actual.vertices[0], actual.faces[0], actual.vertex_colors[0])
        for left, right in zip(fields, expected, strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        receipt["cells"][key] = {
            "bit_exact": True,
            "vertices": actual.vertices.shape[1],
            "faces": actual.faces.shape[1],
        }
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(key, receipt["cells"][key], flush=True)


if __name__ == "__main__":
    main()
