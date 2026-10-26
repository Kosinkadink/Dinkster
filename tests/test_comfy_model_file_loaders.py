"""Torch-free schemas for native bound model-file loaders."""

import subprocess
import sys

from dinkster_api.v1 import ASSET_TYPE, AssetWidget, TypeExpr
from dinkster_nodes_comfy.model_file_loaders import MODEL_FILE_LOADER_NODES


def test_model_file_loaders_use_bound_asset_inputs_and_expected_outputs() -> None:
    schemas = {node.schema().node_type: node.schema() for node in MODEL_FILE_LOADER_NODES}

    assert tuple(schemas) == (
        "CheckpointLoaderSimple",
        "CheckpointLoader",
        "unCLIPCheckpointLoader",
        "VAELoader",
        "UNETLoader",
        "CLIPLoader",
        "DualCLIPLoader",
        "CLIPVisionLoader",
    )
    assert tuple(output.id for output in schemas["CheckpointLoaderSimple"].outputs) == (
        "MODEL",
        "CLIP",
        "VAE",
    )
    assert tuple(output.id for output in schemas["unCLIPCheckpointLoader"].outputs) == (
        "MODEL",
        "CLIP",
        "VAE",
        "CLIP_VISION",
    )
    assert tuple(input.id for input in schemas["DualCLIPLoader"].inputs) == (
        "clip_name1",
        "clip_name2",
        "type",
        "device",
    )
    for schema in schemas.values():
        for input in schema.inputs:
            if input.id.endswith("name"):
                assert input.type == TypeExpr.concrete(ASSET_TYPE)
                assert isinstance(input.widget, AssetWidget)


def test_model_file_loader_defaults_match_comfyui() -> None:
    schemas = {node.schema().node_type: node.schema() for node in MODEL_FILE_LOADER_NODES}
    assert schemas["UNETLoader"].inputs[1].default == "default"
    assert schemas["CLIPLoader"].inputs[1].default == "stable_diffusion"
    assert schemas["CLIPLoader"].inputs[2].default == "default"
    assert schemas["DualCLIPLoader"].inputs[2].default == "sdxl"
    assert schemas["DualCLIPLoader"].inputs[3].default == "default"


def test_model_file_loader_schema_discovery_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
from dinkster_nodes_comfy.model_file_loaders import MODEL_FILE_LOADER_NODES
assert len(MODEL_FILE_LOADER_NODES) == 8
assert sys.modules['torch'] is None
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
