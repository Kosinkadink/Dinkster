"""Torch-free public frame interpolation loader contracts."""

import subprocess
import sys

from dinkster_api.v1 import (
    ASSET_TYPE,
    CORE_INT,
    IMAGE_TYPE,
    INTERP_MODEL_TYPE,
    AssetWidget,
    NumberWidget,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.interpolation import FrameInterpolate
from dinkster_nodes_comfy.interpolation_loader import FrameInterpolationModelLoader
from dinkster_workers import load_manifest


def test_interpolation_loader_schema_and_registration() -> None:
    schema = FrameInterpolationModelLoader.schema()
    assert schema.node_type == "FrameInterpolationModelLoader"
    assert schema.display_name == "Load Frame Interpolation Model"
    assert schema.category == "model/loaders"
    assert [(item.id, item.type.types) for item in schema.inputs] == [("model_name", (ASSET_TYPE,))]
    assert schema.inputs[0].widget == AssetWidget(
        accept=("application/octet-stream",), kind="model/frame-interpolation"
    )
    assert [(item.id, item.type.types) for item in schema.outputs] == [
        ("INTERP_MODEL", (INTERP_MODEL_TYPE,))
    ]
    assert COMFY_NODES.count(FrameInterpolationModelLoader) == 1
    consumer = FrameInterpolate.schema()
    assert consumer.display_name == "Run Frame Interpolation Model"
    assert consumer.category == "video"
    assert [(item.id, item.type.types) for item in consumer.inputs] == [
        ("interp_model", (INTERP_MODEL_TYPE,)),
        ("images", (IMAGE_TYPE,)),
        ("multiplier", (CORE_INT,)),
    ]
    assert consumer.inputs[2].default == 2
    assert consumer.inputs[2].widget == NumberWidget(min=2, max=16)
    assert [(item.id, item.type.types) for item in consumer.outputs] == [("IMAGE", (IMAGE_TYPE,))]
    assert COMFY_NODES.count(FrameInterpolate) == 1
    assert (
        tuple(node.schema().node_type for node in COMFY_NODES)
        == load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml").namespaces
    )


def test_interpolation_loader_cold_import_does_not_load_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dinkster_nodes_comfy; "
            "from dinkster_nodes_comfy.interpolation_loader import FrameInterpolationModelLoader; "
            "FrameInterpolationModelLoader.schema(); "
            "assert 'torch' not in sys.modules; "
            "assert not any(k.startswith('dinkster_inference.') for k in sys.modules)",
        ],
        check=True,
    )
