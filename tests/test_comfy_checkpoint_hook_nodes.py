"""Cold checkpoint-hook schemas and native pack registration."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dinkster_api.v1 import ASSET_TYPE, CORE_FLOAT, HOOKS_TYPE, AssetWidget, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.hook_checkpoint import HOOK_CHECKPOINT_NODES


def test_checkpoint_hook_schemas_preserve_file_slot_strengths_and_optional_hooks() -> None:
    for node, names, display in zip(
        HOOK_CHECKPOINT_NODES,
        (
            ("ckpt_name", "strength_model", "strength_clip", "prev_hooks"),
            ("ckpt_name", "strength_model", "prev_hooks"),
        ),
        ("Create Hook Model as LoRA", "Create Hook Model as LoRA (MO)"),
        strict=True,
    ):
        schema = node.schema()
        assert schema.node_type == node.__name__
        assert node in COMFY_NODES
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


def test_checkpoint_hooks_serialize_without_torch_and_match_manifest_order() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads(
        (root / "packages/dinkster-nodes-comfy/dinkster-pack.toml").read_text()
    )
    assert tuple(manifest["pack"]["namespaces"]) == (
        "devices",
        *(node.schema().node_type for node in COMFY_NODES),
    )
