"""Cold checkpoint-hook schemas and native pack registration."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dinkster_api.v1 import ASSET_TYPE, CORE_FLOAT, HOOKS_TYPE, AssetWidget, NumberWidget, TypeExpr
from dinkster_models import NODES
from dinkster_models.hook_checkpoint import (
    CreateHookModelAsLora,
    CreateHookModelAsLoraModelOnly,
)


def test_checkpoint_hook_schemas_preserve_file_slot_strengths_and_optional_hooks() -> None:
    for node, node_type, names, display in zip(
        (CreateHookModelAsLora, CreateHookModelAsLoraModelOnly),
        (
            "dinkster.models.create_hook_model_as_lora",
            "dinkster.models.create_hook_model_as_lora_model_only",
        ),
        (
            ("ckpt_name", "strength_model", "strength_clip", "prev_hooks"),
            ("ckpt_name", "strength_model", "prev_hooks"),
        ),
        ("Create Hook Model as LoRA", "Create Hook Model as LoRA (MO)"),
        strict=True,
    ):
        schema = node.schema()
        assert schema.node_type == node_type
        assert node in NODES
        assert schema.display_name == display
        assert schema.category == "advanced/hooks/create"
        assert tuple(item.id for item in schema.inputs) == names
        file = schema.inputs[0]
        assert file.type == TypeExpr.concrete(ASSET_TYPE)
        assert file.required and file.default is None
        assert file.widget == AssetWidget(
            kind="model/checkpoint", accept=("application/octet-stream",)
        )
        for strength in schema.inputs[1:-1]:
            assert strength.type == TypeExpr.concrete(CORE_FLOAT)
            assert strength.required and strength.default == 1.0
            assert strength.widget == NumberWidget(min=-20.0, max=20.0, step=0.01)
        previous = schema.inputs[-1]
        assert previous.type == TypeExpr.concrete(HOOKS_TYPE)
        assert not previous.required and previous.default is None
        assert [(item.id, item.type) for item in schema.outputs] == [
            ("HOOKS", TypeExpr.concrete(HOOKS_TYPE))
        ]


def test_checkpoint_hooks_match_pack_namespace_and_entry_order() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads(
        (root / "packages/dinkster-nodes-models/dinkster-pack.toml").read_text()
    )
    assert manifest["pack"]["namespaces"] == ["dinkster.models", "devices"]
    start = NODES.index(CreateHookModelAsLora)
    assert NODES[start : start + 2] == (CreateHookModelAsLora, CreateHookModelAsLoraModelOnly)
