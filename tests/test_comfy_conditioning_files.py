"""Torch-free native conditioning file contracts."""

import subprocess
import sys

from dinkster_api.v1 import ASSET_TYPE, CONDITIONING_TYPE, AssetWidget, MountWidget
from dinkster_conditioning import NODES
from dinkster_conditioning.conditioning_files import ConditioningLoader, SaveConditioning
from dinkster_schema import claim_covers
from dinkster_workers import load_manifest


def test_conditioning_file_schemas_and_registration() -> None:
    loader = ConditioningLoader.schema()
    assert loader.node_type == "dinkster.conditioning.conditioning_loader"
    assert loader.display_name == "Load Conditioning"
    assert loader.category == "model/loaders"
    assert [(item.id, item.type.types) for item in loader.inputs] == [
        ("conditioning_name", (ASSET_TYPE,))
    ]
    assert loader.inputs[0].widget == AssetWidget(
        accept=("application/octet-stream",), kind="model/embedding"
    )
    assert [(item.id, item.type.types) for item in loader.outputs] == [
        ("CONDITIONING", (CONDITIONING_TYPE,))
    ]
    saver = SaveConditioning.schema()
    assert saver.node_type == "dinkster.conditioning.save_conditioning"
    assert [item.id for item in saver.inputs] == ["conditioning", "filename_prefix", "mount"]
    assert saver.inputs[0].type.types == (CONDITIONING_TYPE,)
    assert saver.inputs[1].default == "conditioning/ComfyUI"
    assert not saver.inputs[2].required and saver.inputs[2].default is None
    assert saver.inputs[2].widget == MountWidget()
    assert saver.inputs[2].adjacent_to == "filename_prefix"
    assert [(item.id, item.type.types) for item in saver.outputs] == [
        ("conditioning", (CONDITIONING_TYPE,))
    ]
    assert saver.output_node and not saver.idempotent
    assert NODES.count(ConditioningLoader) == NODES.count(SaveConditioning) == 1
    manifest = load_manifest("packages/dinkster-nodes-conditioning/dinkster-pack.toml")
    assert all(
        any(claim_covers(claim, node.schema().node_type) for claim in manifest.namespaces)
        for node in NODES
    )


def test_conditioning_file_import_is_torch_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dinkster_conditioning; "
            "from dinkster_conditioning.conditioning_files import "
            "ConditioningLoader, SaveConditioning; "
            "ConditioningLoader.schema(); SaveConditioning.schema(); "
            "assert 'torch' not in sys.modules; "
            "assert not any(k.startswith('dinkster_inference.') for k in sys.modules)",
        ],
        check=True,
    )
