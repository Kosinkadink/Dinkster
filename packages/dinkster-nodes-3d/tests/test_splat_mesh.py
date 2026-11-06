"""Surface extraction, variable-length batches and the frozen wire schema."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from dinkster_api.v1 import initialize_inference_process, schema_to_wire

from tools.compare_nodes import compare

SplatToMesh = importlib.import_module("dinkster_nodes_3d.splat_mesh").SplatToMesh


def test_splat_mesh_schema_matches_reference_except_frozen_id() -> None:
    reference = json.loads(
        (Path(__file__).parents[3] / "tools/data/comfy_core_reference.json").read_text()
    )
    row = next(item for item in reference["nodes"] if item["class_name"] == "SplatToMesh")
    schema = SplatToMesh.define_schema()
    assert schema.node_type == "dinkster.3d.splat_to_mesh"
    assert "SplatToMesh" in schema.search_terms
    assert schema.aliases == ()
    wire = schema_to_wire(schema)
    wire["nodeType"] = "SplatToMesh"
    assert compare(row, wire) == []


def test_splat_mesh_counts_filter_padding_and_preserve_linear_color() -> None:
    initialize_inference_process()
    torch = importlib.import_module("torch")
    positions = torch.tensor([[[0.3, 0.5, -0.7], [20, 20, 20]]] * 2)
    splat = {
        "positions": positions,
        "scales": torch.full_like(positions, 0.2),
        "rotations": torch.tensor([[[1.0, 0, 0, 0]]] * 2).expand(2, 2, 4),
        "opacities": torch.tensor([[[1.0], [1.0]], [[0.001], [1.0]]]),
        "sh": torch.tensor(
            [[[[(0.8 - 0.5) / 0.28209479177387814, 0.0, -0.5 / 0.28209479177387814]]]] * 2
        ).expand(2, 2, 1, 3),
        "counts": torch.tensor([1, 1]),
    }
    mesh: Any = SplatToMesh.execute(
        splat=splat,
        resolution=64,
        kernel=8,
        min_component=0,
        min_opacity=0.02,
        color_sharpen=1.0,
        smooth=1,
    )["mesh"]
    assert mesh.unlit is True
    assert int(mesh.vertex_counts[0]) > 0 and int(mesh.face_counts[0]) > 0
    assert int(mesh.vertex_counts[1]) == 0 and int(mesh.face_counts[1]) == 0
    verts = mesh.vertices[0, : mesh.vertex_counts[0]].numpy()
    # The output rotates 180 degrees about X, and padding must not affect the extent.
    np.testing.assert_allclose(
        (verts.min(0) + verts.max(0)) / 2, [0.3, -0.5, 0.7], atol=0.03, rtol=0
    )
    assert np.max(np.abs(verts)) < 2
    faces = mesh.faces[0, : mesh.face_counts[0]]
    assert int(faces.min()) >= 0 and int(faces.max()) < int(mesh.vertex_counts[0])
    color = mesh.vertex_colors[0, : mesh.vertex_counts[0]].numpy()
    expected = [((0.8 + 0.055) / 1.055) ** 2.4, ((0.5 + 0.055) / 1.055) ** 2.4, 0]
    np.testing.assert_allclose(color, np.broadcast_to(expected, color.shape), atol=0.001, rtol=0)
    removed: Any = SplatToMesh.execute(splat=splat, resolution=64, kernel=8, min_component=100000)[
        "mesh"
    ]
    assert removed.vertex_counts.tolist() == [0, 0]
    assert removed.face_counts.tolist() == [0, 0]
