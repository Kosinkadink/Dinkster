"""DA3 public results, bound files and ordinary pack-host transport on CPU."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    DA3_GEOMETRY_TYPE,
    DA3_MODEL_TYPE,
    TypeRegistry,
)
from dinkster_vision_depth_anything_3 import register_types
from dinkster_vision_depth_anything_3.depth_anything_3 import (
    DA3GeometryToMesh,
    DA3Render,
)


def test_pack_registers_only_its_vision_packet_and_preserves_existing_authority() -> None:
    registry = TypeRegistry()
    resident = registry.register(DA3_MODEL_TYPE)
    register_types(registry)
    assert {"DA3_GEOMETRY", "DA3_MODEL", "IMAGE", "MASK", "MESH", "ASSET"} <= set(
        registry.type_ids()
    )
    assert not {
        "FACE_LANDMARKS",
        "MHR_POSE_DATA",
        "KIMODO_POSE_DATA",
        "FACE_DETECTION_MODEL",
        "SAM3D_BODY_MODEL",
    } & set(registry.type_ids())
    packet = registry.spec(DA3_GEOMETRY_TYPE)
    register_types(registry)
    assert registry.spec(DA3_GEOMETRY_TYPE) is packet
    assert registry.spec(DA3_MODEL_TYPE) is resident


def _extrinsics() -> Any:
    return torch.tensor(
        [
            [[1.0, 0, 0, 7], [0, 1, 0, 11], [0, 0, 1, 13]],
            [[0.0, -1, 0, 3], [1, 0, 0, -5], [0, 0, 1, 2]],
        ]
    )[None]


def _pixels() -> Any:
    return (
        torch.tensor([[0.0, 0.2, 0.4], [0.5, 0.7, 0.9]])[:, None, None].expand(2, 3, 5, 3).clone()
    )


def _packet() -> dict[str, Any]:
    return {
        "depth": torch.full((2, 3, 5), 4.0),
        "image": _pixels(),
        "intrinsics": torch.tensor([[2.0, 0, 1], [0, 4, 1], [0, 0, 1]])[None, None].repeat(
            1, 2, 1, 1
        ),
        "confidence": torch.full((2, 3, 5), 7.0),
        "sky": torch.zeros(2, 3, 5),
        "mode": "multiview",
    }


@pytest.mark.parametrize("camera", (False, True))
def test_render_and_mesh_values_preserve_source(camera: bool) -> None:
    packet = _packet()
    if camera:
        packet["extrinsics"] = _extrinsics()
    before = {
        key: value.clone() for key, value in packet.items() if isinstance(value, torch.Tensor)
    }
    raw = DA3Render.execute(
        da3_geometry=packet,
        output="depth",
        **{
            "output.normalization": "raw",
            "output.apply_sky_clip": False,
        },
    )["IMAGE"]
    torch.testing.assert_close(raw, torch.full((2, 3, 5, 3), 4.0), rtol=0, atol=0)
    confidence = DA3Render.execute(
        da3_geometry=packet,
        output="confidence",
        **{
            "output.colored": False,
        },
    )["IMAGE"]
    torch.testing.assert_close(confidence, torch.ones(2, 3, 5, 3), rtol=0, atol=0)
    mesh = DA3GeometryToMesh.execute(
        da3_geometry=packet,
        batch_index=1,
        decimation=2,
        discontinuity_threshold=0,
    )["MESH"]
    torch.testing.assert_close(
        mesh.vertices,
        torch.tensor(
            [
                [
                    [4.0, -5.0, -2.0],
                    [4.0, -1.0, -2.0],
                    [4.0, 3.0, -2.0],
                    [6.0, -5.0, -2.0],
                    [6.0, -1.0, -2.0],
                    [6.0, 3.0, -2.0],
                ]
                if camera
                else [
                    [-2.0, 1.0, -4.0],
                    [2.0, 1.0, -4.0],
                    [6.0, 1.0, -4.0],
                    [-2.0, -1.0, -4.0],
                    [2.0, -1.0, -4.0],
                    [6.0, -1.0, -4.0],
                ]
            ]
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        mesh.faces, torch.tensor([[[0, 4, 1], [1, 5, 2], [0, 3, 4], [1, 4, 5]]])
    )
    torch.testing.assert_close(mesh.texture, before["image"][1:2], rtol=0, atol=0)
    for key, value in before.items():
        torch.testing.assert_close(packet[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="empty mesh"):
        DA3GeometryToMesh.execute(da3_geometry=packet | {"sky": torch.ones(2, 3, 5)})
