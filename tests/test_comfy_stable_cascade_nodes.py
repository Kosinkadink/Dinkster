"""Torch-free Stable Cascade public interfaces."""

from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.stable_cascade import STABLE_CASCADE_NODES


def test_stable_cascade_interfaces() -> None:
    expected = (
        (
            "StableCascade_EmptyLatentImage",
            ("width", "height", "compression", "batch_size"),
            ("stage_c", "stage_b"),
        ),
        ("StableCascade_StageB_Conditioning", ("conditioning", "stage_c"), ("CONDITIONING",)),
        ("StableCascade_StageC_VAEEncode", ("image", "vae", "compression"), ("stage_c", "stage_b")),
        (
            "StableCascade_SuperResolutionControlnet",
            ("image", "vae"),
            ("controlnet_input", "stage_c", "stage_b"),
        ),
    )
    nodes = STABLE_CASCADE_NODES
    assert (
        tuple(
            (
                node.schema().node_type,
                tuple(i.id for i in node.schema().inputs),
                tuple(o.id for o in node.schema().outputs),
            )
            for node in nodes
        )
        == expected
    )
    for node in nodes:
        assert node in COMFY_NODES
        assert schema_to_wire(node.schema())
    compression = STABLE_CASCADE_NODES[0].schema().inputs[2]
    assert compression.default == 42
    assert compression.advanced
