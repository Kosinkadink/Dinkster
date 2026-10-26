"""Encoder export schemas retain their source inputs and mount layout."""

from dinkster_api.v1 import CLIP_TYPE, VAE_TYPE, MountWidget, TypeExpr
from dinkster_nodes_comfy.model_file_savers import CLIPSave, VAESave


def test_encoder_savers_keep_model_inputs_and_append_the_public_mount() -> None:
    for node, value_type, input_name, prefix in (
        (CLIPSave, CLIP_TYPE, "clip", "clip/ComfyUI"),
        (VAESave, VAE_TYPE, "vae", "vae/ComfyUI_vae"),
    ):
        schema = node.schema()
        assert [item.id for item in schema.inputs] == [input_name, "filename_prefix", "mount"]
        assert schema.inputs[0].type == TypeExpr.concrete(value_type)
        assert schema.inputs[1].default == prefix
        assert schema.inputs[2].widget == MountWidget()
        assert schema.inputs[2].adjacent_to == "filename_prefix"
        assert schema.inputs[2].default is None and not schema.inputs[2].required
        assert schema.outputs == ()
        assert schema.output_node and not schema.idempotent
