"""Cold LoRA-hook schemas and native pack registration."""

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_api.v1 import ASSET_TYPE, CORE_FLOAT, HOOKS_TYPE, AssetWidget, NumberWidget, TypeExpr
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.hook_lora import HOOK_LORA_NODES


def test_lora_hook_schemas_and_native_registration() -> None:
    for node, strengths in zip(
        HOOK_LORA_NODES, (("strength_model", "strength_clip"), ("strength_model",)), strict=True
    ):
        schema = node.schema()
        assert node in COMFY_NODES
        assert schema.node_type == node.__name__
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
        (root / "packages/dinkster-nodes-comfy/dinkster-pack.toml").read_text()
    )
    assert tuple(manifest["pack"]["namespaces"]) == tuple(
        node.schema().node_type for node in COMFY_NODES
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_schema import schema_to_wire; "
            "from dinkster_nodes_comfy.hook_lora import HOOK_LORA_NODES; "
            "assert all(schema_to_wire(node.schema()) for node in HOOK_LORA_NODES); "
            "assert 'torch' not in sys.modules; "
            "assert not any(name.startswith('dinkster_inference.') for name in sys.modules)",
        ],
        cwd=root,
        check=True,
    )
