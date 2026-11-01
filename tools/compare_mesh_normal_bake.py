"""Append new CPU normal-map bake cells against the immutable issue 511 source."""

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
from dinkster_nodes_comfy.mesh_bake import BakeNormalMapFromMesh

REFERENCE = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
SOURCES = {
    "source": "3b7083984c974d9bc5d14e026a37d0eee0a4063b61a7df20ae15bdd68015a141",
    "qem": "18fe57a83ec8de3f522d76407ea5fcb2ce1576fb21b5994016a88947ad2c5b7e",
    "remesh": "2444edfce3f86c3b7fdd76363595437ff745a5e3883953cc235d50225dfa6e23",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*SOURCES, "receipt"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    assert management.get_torch_device().type == "cpu"
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
    names = {
        "source": {
            "_rasterize_uv_barycentric",
            "_msb_int64",
            "_morton_expand21",
            "_build_triangle_bvh",
            "_closest_points_on_mesh_bvh",
            "_ray_tri_intersect",
            "_closest_hit_rays_bvh",
            "_compute_vertex_tangents",
            "_vertex_tangents_for_item",
            "_barycentric",
            "_bake_normal_map",
            "_jfa_fill_gpu",
            "_normalize_uvs_to_unit",
            "BakeNormalMapFromMesh",
        },
        "qem": {"_compute_vertex_normals"},
        "remesh": {"_point_tri_closest"},
    }
    for name, digest in SOURCES.items():
        path = getattr(args, name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        definitions = [
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names[name]
        ]
        assert len(definitions) == len(names[name])
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    receipt = (
        json.loads(args.receipt.read_text())
        if args.receipt.exists()
        else dict(reference=REFERENCE, sources=SOURCES, cells={})
    )
    assert receipt["reference"] == REFERENCE and receipt["sources"] == SOURCES
    for case in (
        "authored",
        "computed",
        "mirrored",
        "uv-fit",
        "fallback",
        "backface",
        "ragged",
        "uncovered",
    ):
        if case in receipt["cells"]:
            continue
        vertices = torch.tensor(
            [[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]]
        )
        faces = torch.tensor([[[0, 1, 2], [0, 2, 3]]])
        uvs = torch.tensor([[[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]]])
        high_vertices = vertices.clone()
        high_vertices[..., 2] = high_vertices[..., 0] * 0.13 + high_vertices[..., 1] * 0.07
        high_faces = faces.clone()
        normals = torch.tensor([[[0.25, -0.15, 0.9]] * 4])
        if case == "mirrored":
            uvs[..., 1] = 1.0 - uvs[..., 1]
        if case == "uv-fit":
            uvs = uvs * 4.0 - 0.7
        if case == "fallback":
            high_vertices[..., 2] += 2.0
        if case == "backface":
            high_faces = high_faces.flip(-1)
            normals[..., 2] = -1.0
        if case == "uncovered":
            uvs = torch.zeros_like(uvs)
        low = TriangleMeshBatch(
            vertices, faces, uvs=uvs, vertex_counts=torch.tensor([4]), face_counts=torch.tensor([2])
        )
        high = TriangleMeshBatch(
            high_vertices,
            high_faces,
            normals=None if case in ("computed", "ragged") else normals,
            vertex_counts=torch.tensor([4]),
            face_counts=torch.tensor([2]),
        )
        if case == "ragged":
            low.vertices, low.faces, low.uvs = (
                vertices.repeat(2, 1, 1),
                faces.repeat(2, 1, 1),
                uvs.repeat(2, 1, 1),
            )
            low.vertex_counts, low.face_counts = torch.tensor([4, 3]), torch.tensor([2, 0])
        kwargs = dict(
            low_poly=low,
            high_poly=high,
            resolution=64,
            cage_distance=0.04 if case == "fallback" else 0.1,
            ignore_backfaces=True,
        )
        expected = namespace["BakeNormalMapFromMesh"].execute(**kwargs)[0]
        actual = BakeNormalMapFromMesh.execute(**kwargs)["IMAGE"]
        assert torch.equal(actual, expected), (case, float((actual - expected).abs().max()))
        receipt["cells"][case] = dict(
            shape=list(actual.shape),
            dtype=str(actual.dtype),
            bit_exact=True,
            sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
        )
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(case, "bit-exact", flush=True)


if __name__ == "__main__":
    main()
