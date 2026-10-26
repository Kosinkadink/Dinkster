"""Torch-free public interfaces for native ComfyUI inpainting nodes."""

from dinkster_api.v1 import schema_to_wire
from dinkster_nodes_comfy.inpaint import INPAINT_NODES


def test_inpaint_schemas_match_core_interfaces() -> None:
    expected = (
        (
            "VAEEncodeForInpaint",
            "VAE Encode (for Inpainting)",
            "model/latent",
            ("pixels", "vae", "mask", "grow_mask_by"),
            ("LATENT",),
        ),
        (
            "InpaintModelConditioning",
            "InpaintModelConditioning",
            "model/conditioning",
            ("positive", "negative", "vae", "pixels", "mask", "noise_mask"),
            ("positive", "negative", "latent"),
        ),
    )
    actual = tuple(
        (
            node.schema().node_type,
            node.schema().display_name,
            node.schema().category,
            tuple(input_spec.id for input_spec in node.schema().inputs),
            tuple(output_spec.id for output_spec in node.schema().outputs),
        )
        for node in INPAINT_NODES
    )
    assert actual == expected
    assert INPAINT_NODES[0].schema().inputs[-1].default == 6
    assert INPAINT_NODES[1].schema().inputs[-1].default is True
    for node in INPAINT_NODES:
        assert schema_to_wire(node.schema())
