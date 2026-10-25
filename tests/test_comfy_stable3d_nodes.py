"""Torch-free Stable Zero123 and Stable Video 3D public interfaces."""

from dinkster_api.v1 import NumberWidget, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.stable3d import STABLE3D_NODES


def test_stable3d_interfaces() -> None:
    expected = (
        (
            "StableZero123_Conditioning",
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
            "StableZero123_Conditioning_Batched",
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
            "SV3D_Conditioning",
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
        assert node in COMFY_NODES
        assert tuple(output.id for output in node.schema().outputs) == (
            "positive",
            "negative",
            "latent",
        )
        assert schema_to_wire(node.schema())
    batched = STABLE3D_NODES[1].schema()
    assert batched.display_name == ""
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
