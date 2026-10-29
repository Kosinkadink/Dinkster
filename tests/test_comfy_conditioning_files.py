"""Torch-free native conditioning file contracts."""

import subprocess
import sys

from dinkster_api.v1 import ASSET_TYPE, CONDITIONING_TYPE, AssetWidget, MountWidget
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.conditioning_files import ConditioningLoader, SaveConditioning
from dinkster_workers import load_manifest


def test_conditioning_file_schemas_and_registration() -> None:
    loader = ConditioningLoader.schema()
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
    assert COMFY_NODES.count(ConditioningLoader) == COMFY_NODES.count(SaveConditioning) == 1
    assert ("devices", *(node.schema().node_type for node in COMFY_NODES)) == load_manifest(
        "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    ).namespaces


def test_conditioning_file_import_is_torch_free() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dinkster_nodes_comfy; "
            "from dinkster_nodes_comfy.conditioning_files import "
            "ConditioningLoader, SaveConditioning; "
            "ConditioningLoader.schema(); SaveConditioning.schema(); "
            "assert 'torch' not in sys.modules; "
            "assert not any(k.startswith('dinkster_inference.') for k in sys.modules)",
        ],
        check=True,
    )
