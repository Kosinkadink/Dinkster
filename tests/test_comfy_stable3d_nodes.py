"""Torch-free Stable Zero123 and Stable Video 3D public interfaces."""

from dinkster_api.v1 import NumberWidget, schema_to_wire
from dinkster_sd import NODES
from dinkster_sd.stable3d import (
    StableZero123_Conditioning,
    StableZero123_Conditioning_Batched,
    SV3D_Conditioning,
)

STABLE3D_NODES = (
    StableZero123_Conditioning,
    StableZero123_Conditioning_Batched,
    SV3D_Conditioning,
)


def test_stable3d_interfaces() -> None:
    expected = (
        (
            "dinkster.sd.stable_zero123_conditioning",
            (
                "clip_vision",
                "init_image",
                "vae",
                "width",
                "height",
                "batch_size",
                "elevation",
                "azimuth",
            ),
        ),
        (
            "dinkster.sd.stable_zero123_conditioning_batched",
            (
                "clip_vision",
                "init_image",
                "vae",
                "width",
                "height",
                "batch_size",
                "elevation",
                "azimuth",
                "elevation_batch_increment",
                "azimuth_batch_increment",
            ),
        ),
        (
            "dinkster.sd.sv3_d_conditioning",
            ("clip_vision", "init_image", "vae", "width", "height", "video_frames", "elevation"),
        ),
    )
    assert (
        tuple(
            (node.schema().node_type, tuple(input_.id for input_ in node.schema().inputs))
            for node in STABLE3D_NODES
        )
        == expected
    )
    for node in STABLE3D_NODES:
        assert NODES.count(node) == 1
        assert tuple(output.id for output in node.schema().outputs) == (
            "positive",
            "negative",
            "latent",
        )
        assert schema_to_wire(node.schema())
    batched = STABLE3D_NODES[1].schema()
    assert batched.display_name == "StableZero123_Conditioning_Batched"
    assert batched.inputs[-2].advanced and batched.inputs[-1].advanced

    angle_names = {
        "elevation",
        "azimuth",
        "elevation_batch_increment",
        "azimuth_batch_increment",
    }
    angles = [
        item for node in STABLE3D_NODES for item in node.schema().inputs if item.id in angle_names
    ]
    assert len(angles) == 7
    assert all(
        isinstance(item.widget, NumberWidget) and item.widget.round is False for item in angles
    )
