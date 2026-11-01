"""Append new CPU voxel texture cells against the immutable issue 511 source."""

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
import scipy.ndimage as ndi
import torch
from dinkster_api.v1 import TriangleMeshBatch, initialize_inference_process
from dinkster_nodes_comfy.voxel_texture import BakeTextureFromVoxel
from scipy.spatial import cKDTree
from tqdm import tqdm

REFERENCE = "bc5f880dbdd43e426077f42ccec53bb577ffee9f"
SOURCES = {
    "source": "3b7083984c974d9bc5d14e026a37d0eee0a4063b61a7df20ae15bdd68015a141",
    "remesh": "2444edfce3f86c3b7fdd76363595437ff745a5e3883953cc235d50225dfa6e23",
}


class Progress:
    def __init__(self, total: int) -> None:
        pass

    def update(self, value: int) -> None:
        pass


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
        ndi=ndi,
        cKDTree=cKDTree,
        tqdm=tqdm,
        math=math,
        logging=logging,
        _BRUTE_NEAREST_MAX=8192,
        comfy=SimpleNamespace(
            model_management=management, utils=SimpleNamespace(ProgressBar=Progress)
        ),
        IO=SimpleNamespace(ComfyNode=object, NodeOutput=lambda *values: values),
        get_mesh_batch_item=lambda mesh, i: (
            mesh.vertices[i, : int(mesh.vertex_counts[i])],
            mesh.faces[i, : int(mesh.face_counts[i])],
        ),
    )
    names = {
        "source": {
            "_mesh_face_count",
            "_prepare_gpu_mesh_processing",
            "_rasterize_uv_barycentric",
            "_interp_vertex_attr",
            "_bake_position_map",
            "_trilinear_sample_sparse",
            "_trilinear_sample_sparse_gpu",
            "_nearest_voxel_sample_gpu",
            "_sample_voxel_attrs_per_texel",
            "_msb_int64",
            "_morton_expand21",
            "_build_triangle_bvh",
            "_closest_points_on_mesh_bvh",
            "_back_project_positions",
            "_jfa_fill_gpu",
            "_seam_fill",
            "_normalize_uvs_to_unit",
            "bake_texture_from_voxel_fn",
            "_mr_channel",
            "BakeTextureFromVoxel",
        },
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
    for case in ("rgb", "pbr", "sparse", "nearest", "reference", "uv-fit", "ragged", "uncovered"):
        if case in receipt["cells"]:
            continue
        vertices = torch.tensor(
            [[[-0.25, -0.25, 0.0], [0.25, -0.25, 0.0], [0.25, 0.25, 0.0], [-0.25, 0.25, 0.0]]]
        )
        faces = torch.tensor([[[0, 1, 2], [0, 2, 3]]])
        uv = torch.tensor([[[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]]])
        coords = torch.cartesian_prod(torch.arange(2), torch.arange(2), torch.arange(2))
        colors = (torch.arange(48).reshape(8, 6).float() * 0.037).remainder(1.3) - 0.1
        if case == "rgb":
            colors = colors[:, :3]
        if case == "sparse":
            coords, colors = coords[[0, 2, 5]], colors[[0, 2, 5]]
        if case == "nearest":
            vertices += 4.0
        if case == "uv-fit":
            uv = uv * 3 - 0.7
        if case == "uncovered":
            uv = torch.zeros_like(uv)
        mesh = TriangleMeshBatch(
            vertices, faces, uvs=uv, vertex_counts=torch.tensor([4]), face_counts=torch.tensor([2])
        )
        reference = None
        if case == "reference":
            reference = TriangleMeshBatch(
                vertices + torch.tensor([0, 0, 0.13]),
                faces,
                vertex_counts=torch.tensor([4]),
                face_counts=torch.tensor([2]),
            )
        if case == "ragged":
            mesh.vertices, mesh.faces, mesh.uvs = (
                vertices.repeat(2, 1, 1),
                faces.repeat(2, 1, 1),
                uv.repeat(2, 1, 1),
            )
            mesh.vertex_counts, mesh.face_counts = torch.tensor([4, 3]), torch.tensor([2, 0])
            coords = torch.cat((torch.zeros((len(coords), 1), dtype=coords.dtype), coords), dim=1)
        voxel = SimpleNamespace(data=coords, voxel_colors=colors, resolution=2)
        kwargs = dict(mesh=mesh, voxel_colors=voxel, texture_size=64, reference_mesh=reference)
        expected = namespace["BakeTextureFromVoxel"].execute(**kwargs)
        actual = BakeTextureFromVoxel.execute(**kwargs)
        values = {}
        for name, wanted in zip(("base_color", "metallic", "roughness"), expected, strict=True):
            value = actual[name]
            assert torch.equal(value, wanted), (case, name, float((value - wanted).abs().max()))
            values[name] = dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                bit_exact=True,
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            )
        receipt["cells"][case] = values
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(case, "bit-exact", flush=True)


if __name__ == "__main__":
    main()
