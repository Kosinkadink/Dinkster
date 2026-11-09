"""Encoder export schemas retain their source inputs and mount layout."""

from dinkster_api.v1 import CLIP_TYPE, MODEL_TYPE, VAE_TYPE, MountWidget, TypeExpr
from dinkster_models import NODES
from dinkster_models.lora_extract_save import LoraSave
from dinkster_models.model_file_savers import CheckpointSave, CLIPSave, ModelSave, VAESave
from dinkster_schema import claim_covers
from dinkster_workers import load_manifest


def test_encoder_savers_keep_model_inputs_and_append_the_public_mount() -> None:
    for node, node_type, value_type, input_name, prefix in (
        (CLIPSave, "dinkster.models.clip_save", CLIP_TYPE, "clip", "clip/ComfyUI"),
        (VAESave, "dinkster.models.vae_save", VAE_TYPE, "vae", "vae/ComfyUI_vae"),
    ):
        schema = node.schema()
        assert schema.node_type == node_type
        assert NODES.count(node) == 1
        assert [item.id for item in schema.inputs] == [input_name, "filename_prefix", "mount"]
        assert schema.inputs[0].type == TypeExpr.concrete(value_type)
        assert schema.inputs[1].default == prefix
        assert schema.inputs[2].widget == MountWidget()
        assert schema.inputs[2].adjacent_to == "filename_prefix"
        assert schema.inputs[2].default is None and not schema.inputs[2].required
        assert schema.outputs == ()
        assert schema.output_node and not schema.idempotent


def test_checkpoint_savers_keep_core_inputs_and_pack_registration() -> None:
    for node, node_type, inputs, types, prefix in (
        (
            CheckpointSave,
            "dinkster.models.checkpoint_save",
            ["model", "clip", "vae"],
            [MODEL_TYPE, CLIP_TYPE, VAE_TYPE],
            "checkpoints/ComfyUI",
        ),
        (
            ModelSave,
            "dinkster.models.model_save",
            ["model"],
            [MODEL_TYPE],
            "diffusion_models/ComfyUI",
        ),
    ):
        schema = node.schema()
        assert schema.node_type == node_type
        assert [item.id for item in schema.inputs] == [*inputs, "filename_prefix", "mount"]
        assert [item.type for item in schema.inputs[:-2]] == list(map(TypeExpr.concrete, types))
        assert all(item.required for item in schema.inputs[:-1])
        assert schema.inputs[-2].default == prefix
        assert schema.inputs[-1].widget == MountWidget()
        assert schema.inputs[-1].adjacent_to == "filename_prefix"
        assert schema.inputs[-1].default is None and not schema.inputs[-1].required
        assert schema.outputs == () and schema.output_node and not schema.idempotent
    manifest = load_manifest("packages/dinkster-nodes-models/dinkster-pack.toml")
    assert all(
        any(claim_covers(claim, node.schema().node_type) for claim in manifest.namespaces)
        for node in NODES
    )
    checkpoint_index = NODES.index(CheckpointSave)
    assert NODES[checkpoint_index : checkpoint_index + 2] == (CheckpointSave, ModelSave)


def test_lora_save_keeps_source_inputs_and_native_manifest_claim() -> None:
    schema = LoraSave.schema()
    assert schema.node_type == "dinkster.models.lora_save"
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
    assert NODES.count(LoraSave) == 1
    manifest = load_manifest("packages/dinkster-nodes-models/dinkster-pack.toml")
    assert any(claim_covers(claim, schema.node_type) for claim in manifest.namespaces)
