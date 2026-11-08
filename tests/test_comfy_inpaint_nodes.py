"""Torch-free public interfaces for native ComfyUI inpainting nodes."""

from dinkster_api.v1 import schema_to_wire
from dinkster_conditioning import NODES as CONDITIONING_NODES
from dinkster_conditioning.inpaint import InpaintModelConditioning
from dinkster_latent import NODES as LATENT_NODES
from dinkster_latent.inpaint import VAEEncodeForInpaint


def test_inpaint_schemas_match_core_interfaces() -> None:
    nodes = (VAEEncodeForInpaint, InpaintModelConditioning)
    expected = (
        (
            "dinkster.latent.vae_encode_for_inpaint",
            "VAE Encode (for Inpainting)",
            "model/latent",
            ("pixels", "vae", "mask", "grow_mask_by"),
            ("LATENT",),
        ),
        (
            "dinkster.conditioning.inpaint_model_conditioning",
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
        for node in nodes
    )
    assert actual == expected
    assert VAEEncodeForInpaint in LATENT_NODES
    assert InpaintModelConditioning in CONDITIONING_NODES
    assert VAEEncodeForInpaint.schema().inputs[-1].default == 6
    assert InpaintModelConditioning.schema().inputs[-1].default is True
    for node in nodes:
        assert schema_to_wire(node.schema())
