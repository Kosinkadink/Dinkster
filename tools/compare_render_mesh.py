"""Append new CPU RenderMesh comparisons against the immutable issue 511 source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from dinkster_api.v1 import TriangleMeshBatch, initialize_inference_process
from dinkster_nodes_comfy.mesh_render import RenderMesh

REFERENCE = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
SOURCE_SHA256 = "3b7083984c974d9bc5d14e026a37d0eee0a4063b61a7df20ae15bdd68015a141"


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
        "_msb_int64",
        "_morton_expand21",
        "_build_triangle_bvh",
        "_ray_tri_intersect",
        "_closest_hit_rays_bvh",
        "_camera_basis",
        "_apply_model_transform",
        "_sample_image01",
        "_render_view",
        "_barycentric",
        "_jfa_fill_gpu",
        "RenderMesh",
    }
    namespace = dict(
        torch=torch,
        np=np,
        math=math,
        logging=logging,
        comfy=SimpleNamespace(model_management=management),
        IO=SimpleNamespace(ComfyNode=object, NodeOutput=lambda *values: values),
        get_mesh_batch_item=lambda mesh, i: (
            mesh.vertices[i, : int(mesh.vertex_counts[i])],
            mesh.faces[i, : int(mesh.face_counts[i])],
        ),
    )
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
        else dict(reference=REFERENCE, source_sha256=SOURCE_SHA256, cells={})
    )
    assert receipt["reference"] == REFERENCE and receipt["source_sha256"] == SOURCE_SHA256
    for camera_kind in ("supplied", "default", "transform"):
        for mode in ("auto", "texture", "vertex colors", "solid", "normal", "depth"):
            key = f"{camera_kind}-{mode}"
            if key in receipt["cells"]:
                continue
            mesh = TriangleMeshBatch(
                torch.tensor(
                    [
                        [
                            [-1.0, -1.0, 0.0],
                            [1.0, -1.0, 0.2],
                            [1.0, 1.0, 0.0],
                            [-1.0, 1.0, -0.1],
                            [100.0, 100.0, 100.0],
                        ]
                    ]
                ),
                torch.tensor([[[0, 1, 2], [0, 2, 3], [4, 4, 4]]]),
                vertex_counts=torch.tensor([4]),
                face_counts=torch.tensor([2]),
                uvs=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]),
                vertex_colors=torch.tensor(
                    [
                        [
                            [0.1, 0.3, 0.8],
                            [0.8, 0.1, 0.3],
                            [0.3, 0.8, 0.1],
                            [0.6, 0.4, 0.2],
                            [1.0, 1.0, 1.0],
                        ]
                    ]
                ),
                texture=torch.tensor(
                    [[[[0.1, 0.2, 0.3], [0.6, 0.4, 0.2]], [[0.8, 0.1, 0.5], [0.2, 0.9, 0.4]]]]
                ),
                normals=torch.tensor([[[0.1, 0.2, 1.0]] * 5])
                if camera_kind == "transform"
                else None,
            )
            camera = (
                None
                if camera_kind == "default"
                else dict(
                    position=dict(x=0.15, y=0.2, z=3.0),
                    target=dict(x=0.0, y=0.0, z=0.0),
                    zoom=1.1,
                    fov=35.0,
                    cameraType="perspective",
                )
            )
            transform = (
                [
                    dict(
                        scale=dict(x=0.8, y=1.1, z=0.6),
                        position=dict(x=0.1, y=-0.2, z=0.05),
                        quaternion=dict(x=0.0, y=0.0, z=0.1, w=math.sqrt(0.99)),
                    )
                ]
                if camera_kind == "transform"
                else None
            )
            kwargs = dict(
                mesh=mesh,
                mode=mode,
                width=80,
                height=64,
                background="#204080",
                camera_info=camera,
                model_3d_info=transform,
            )
            expected = namespace["RenderMesh"].execute(**kwargs)
            actual = RenderMesh.execute(**kwargs)
            for name, wanted in zip(("IMAGE", "MASK"), expected, strict=True):
                torch.testing.assert_close(actual[name], wanted, rtol=0, atol=0)
            receipt["cells"][key] = {"bit_exact": True}
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            print(key, "bit-exact", flush=True)


if __name__ == "__main__":
    main()
