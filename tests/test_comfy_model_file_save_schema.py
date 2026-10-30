"""Encoder export schemas retain their source inputs and mount layout."""

from dinkster_api.v1 import CLIP_TYPE, MODEL_TYPE, VAE_TYPE, MountWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.lora_extract_save import LoraSave
from dinkster_nodes_comfy.model_file_savers import CheckpointSave, CLIPSave, ModelSave, VAESave
from dinkster_workers import load_manifest


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


def test_checkpoint_savers_keep_core_inputs_and_manifest_order() -> None:
    for node, inputs, types, prefix in (
        (
            CheckpointSave,
            ["model", "clip", "vae"],
            [MODEL_TYPE, CLIP_TYPE, VAE_TYPE],
            "checkpoints/ComfyUI",
        ),
        (ModelSave, ["model"], [MODEL_TYPE], "diffusion_models/ComfyUI"),
    ):
        schema = node.schema()
        assert [item.id for item in schema.inputs] == [*inputs, "filename_prefix", "mount"]
        assert [item.type for item in schema.inputs[:-2]] == list(map(TypeExpr.concrete, types))
        assert all(item.required for item in schema.inputs[:-1])
        assert schema.inputs[-2].default == prefix
        assert schema.inputs[-1].widget == MountWidget()
        assert schema.inputs[-1].adjacent_to == "filename_prefix"
        assert schema.inputs[-1].default is None and not schema.inputs[-1].required
        assert schema.outputs == () and schema.output_node and not schema.idempotent
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert ("devices", *(node.schema().node_type for node in COMFY_NODES)) == manifest.namespaces
    checkpoint_index = manifest.namespaces.index("CheckpointSave")
    assert manifest.namespaces[checkpoint_index : checkpoint_index + 2] == (
        "CheckpointSave",
        "ModelSave",
    )


def test_lora_save_keeps_source_inputs_and_native_manifest_claim() -> None:
    schema = LoraSave.schema()
    assert [item.id for item in schema.inputs] == [
        "filename_prefix",
        "rank",
        "lora_type",
        "bias_diff",
        "model_diff",
        "text_encoder_diff",
        "mount",
    ]
    assert schema.inputs[0].default == "loras/ComfyUI_extracted_lora"
    assert schema.inputs[4].type == TypeExpr.concrete(MODEL_TYPE)
    assert schema.inputs[5].type == TypeExpr.concrete(CLIP_TYPE)
    assert not schema.inputs[4].required and not schema.inputs[5].required
    assert schema.inputs[6].widget == MountWidget()
    assert schema.inputs[6].adjacent_to == "filename_prefix"
    assert not schema.inputs[6].required and schema.inputs[6].default is None
    assert schema.outputs == () and schema.output_node and not schema.idempotent
    assert LoraSave in COMFY_NODES
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert ("devices", *(node.schema().node_type for node in COMFY_NODES)) == manifest.namespaces
