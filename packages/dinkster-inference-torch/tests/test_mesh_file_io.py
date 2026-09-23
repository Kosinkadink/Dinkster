from __future__ import annotations

import base64
import json
import struct

import pytest
import torch
from dinkster_inference_torch.gltf_read import parse_container, read_accessor
from dinkster_inference_torch.mesh_file_io import parse_mesh_file


def test_obj_preserves_asymmetric_geometry_uvs_and_colors() -> None:
    obj = b"""\
v 0 0 0 1 0 0
v 2 0 0 0 0.5 0
v 2 3 0 0 0 1
v 0 3 0 1 1 1
vt 0.1 0.2
vt 0.8 0.3
vt 0.9 0.7
vt 0.2 0.9
vn 0 0 2
f -4/1/1 -3/2/1 -2/3/1 -1/4/1
"""

    mesh = parse_mesh_file(obj, "obj")

    assert torch.equal(
        mesh.vertices,
        torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 3.0, 0.0]]]),
    )
    assert torch.equal(mesh.faces, torch.tensor([[[0, 1, 2], [0, 2, 3]]]))
    assert mesh.uvs is not None
    assert torch.allclose(
        mesh.uvs,
        torch.tensor([[[0.1, 0.8], [0.8, 0.7], [0.9, 0.3], [0.2, 0.1]]]),
    )
    assert mesh.vertex_colors is not None
    assert mesh.vertex_colors[0, 1, 1].item() == pytest.approx(0.21404114)
    assert mesh.normals is not None
    assert torch.equal(mesh.normals, torch.tensor([[[0.0, 0.0, 1.0]]] * 4).reshape(1, 4, 3))


def _embedded_gltf(*, glb: bool) -> bytes:
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0)
    indices = struct.pack("<3H", 0, 1, 2)
    payload = positions + indices
    document: dict[str, object] = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(payload)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "nodes": [{"mesh": 0, "translation": [5.0, 7.0, 11.0], "scale": [-1.0, 2.0, 1.0]}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if not glb:
        document["buffers"] = [
            {
                "byteLength": len(payload),
                "uri": "data:application/octet-stream;base64," + base64.b64encode(payload).decode(),
            }
        ]
        return json.dumps(document).encode()

    json_chunk = json.dumps(document, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    payload += b"\0" * (-len(payload) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(payload)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(payload), 0x004E4942)
        + payload
    )


@pytest.mark.parametrize(("format_name", "glb"), [("gltf", False), ("glb", True)])
def test_gltf_applies_scene_transform_and_corrects_reflected_winding(
    format_name: str, glb: bool
) -> None:
    mesh = parse_mesh_file(_embedded_gltf(glb=glb), format_name)

    assert torch.equal(
        mesh.vertices,
        torch.tensor([[[5.0, 7.0, 11.0], [3.0, 7.0, 11.0], [5.0, 13.0, 11.0]]]),
    )
    assert torch.equal(mesh.faces, torch.tensor([[[2, 1, 0]]]))


def test_glb_rejects_declared_lengths_beyond_the_file() -> None:
    glb = bytearray(_embedded_gltf(glb=True))
    struct.pack_into("<I", glb, 12, len(glb))

    with pytest.raises(ValueError, match="truncated in chunk payload"):
        parse_container(bytes(glb))


def test_gltf_accessor_cannot_read_past_its_buffer_view() -> None:
    gltf = {
        "bufferViews": [{"buffer": 0, "byteLength": 8}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "SCALAR"}],
    }

    with pytest.raises(ValueError, match="extends beyond its bufferView"):
        read_accessor(gltf, [struct.pack("<3f", 2.0, 3.0, 5.0)], 0)


def test_gltf_sparse_accessor_rejects_out_of_range_indices() -> None:
    gltf = {
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 1},
            {"buffer": 0, "byteOffset": 1, "byteLength": 4},
        ],
        "accessors": [
            {
                "componentType": 5126,
                "count": 2,
                "type": "SCALAR",
                "sparse": {
                    "count": 1,
                    "indices": {"bufferView": 0, "componentType": 5121},
                    "values": {"bufferView": 1},
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="invalid sparse indices"):
        read_accessor(gltf, [b"\x02" + struct.pack("<f", 7.0)], 0)


def test_binary_stl_uses_facet_normals_and_magics_colors() -> None:
    header = bytearray(80)
    header[:10] = b"COLOR=\x40\x80\xff\xff"
    record = struct.pack(
        "<12fH",
        0.0,
        0.0,
        4.0,
        0.0,
        0.0,
        0.0,
        4.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0x8000,
    )
    mesh = parse_mesh_file(bytes(header) + struct.pack("<I", 1) + record, "stl")

    assert torch.equal(mesh.faces, torch.tensor([[[0, 1, 2]]]))
    assert mesh.normals is not None
    assert torch.equal(mesh.normals, torch.tensor([[[0.0, 0.0, 1.0]]] * 3).reshape(1, 3, 3))
    assert mesh.vertex_colors is not None
    assert mesh.vertex_colors[0, 0].tolist() == pytest.approx(
        [0.05126946, 0.21586053, 1.0], abs=1e-7
    )
