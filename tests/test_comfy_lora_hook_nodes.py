"""Cold LoRA-hook schemas and native pack registration."""

import tomllib
from pathlib import Path

from dinkster_api.v1 import ASSET_TYPE, CORE_FLOAT, HOOKS_TYPE, AssetWidget, NumberWidget, TypeExpr
from dinkster_models import NODES
from dinkster_models.hook_lora import CreateHookLora, CreateHookLoraModelOnly


def test_lora_hook_schemas_and_native_registration() -> None:
    for node, node_type, strengths in zip(
        (CreateHookLora, CreateHookLoraModelOnly),
        ("dinkster.models.create_hook_lora", "dinkster.models.create_hook_lora_model_only"),
        (("strength_model", "strength_clip"), ("strength_model",)),
        strict=True,
    ):
        schema = node.schema()
        assert node in NODES
        assert schema.node_type == node_type
        assert schema.category == "advanced/hooks/create"
        assert tuple(item.id for item in schema.inputs) == ("lora_name", *strengths, "prev_hooks")
        asset = schema.inputs[0]
        assert asset.type == TypeExpr.concrete(ASSET_TYPE)
        assert asset.required and asset.default is None
        assert asset.widget == AssetWidget(kind="model/lora", accept=("application/octet-stream",))
        for strength in schema.inputs[1:-1]:
            assert strength.type == TypeExpr.concrete(CORE_FLOAT)
            assert strength.default == 1.0 and strength.required
            assert strength.widget == NumberWidget(min=-20.0, max=20.0, step=0.01)
        assert not schema.inputs[-1].required
        assert schema.inputs[-1].type == TypeExpr.concrete(HOOKS_TYPE)
        assert [(item.id, item.type) for item in schema.outputs] == [
            ("HOOKS", TypeExpr.concrete(HOOKS_TYPE))
        ]

    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads(
        (root / "packages/dinkster-nodes-models/dinkster-pack.toml").read_text()
    )
    assert manifest["pack"]["namespaces"] == ["dinkster.models", "devices"]
    start = NODES.index(CreateHookLora)
    assert NODES[start : start + 2] == (CreateHookLora, CreateHookLoraModelOnly)
