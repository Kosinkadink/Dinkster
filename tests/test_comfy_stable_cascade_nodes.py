"""Torch-free Stable Cascade public interfaces."""

from dinkster_api.v1 import schema_to_wire
from dinkster_sd import NODES
from dinkster_sd.stable_cascade import (
    StableCascade_EmptyLatentImage,
    StableCascade_StageB_Conditioning,
    StableCascade_StageC_VAEEncode,
    StableCascade_SuperResolutionControlnet,
)


def test_stable_cascade_interfaces() -> None:
    expected = (
        (
            "dinkster.sd.stable_cascade_empty_latent_image",
            ("width", "height", "compression", "batch_size"),
            ("stage_c", "stage_b"),
        ),
        (
            "dinkster.sd.stable_cascade_stage_b_conditioning",
            ("conditioning", "stage_c"),
            ("CONDITIONING",),
        ),
        (
            "dinkster.sd.stable_cascade_stage_c_vae_encode",
            ("image", "vae", "compression"),
            ("stage_c", "stage_b"),
        ),
        (
            "dinkster.sd.stable_cascade_super_resolution_controlnet",
            ("image", "vae"),
            ("controlnet_input", "stage_c", "stage_b"),
        ),
    )
    nodes = (
        StableCascade_EmptyLatentImage,
        StableCascade_StageB_Conditioning,
        StableCascade_StageC_VAEEncode,
        StableCascade_SuperResolutionControlnet,
    )
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
        assert NODES.count(node) == 1
        assert schema_to_wire(node.schema())
    compression = StableCascade_EmptyLatentImage.schema().inputs[2]
    assert compression.default == 42
    assert compression.advanced
