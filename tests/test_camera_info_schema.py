"""Cold public contracts for world-space camera construction."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from dinkster_api.v1 import (
    CAMERA_INFO_STATE_TYPE,
    LOAD3D_CAMERA_TYPE,
    schema_from_wire,
    schema_to_wire,
)
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.camera_info import CreateCameraInfo


def test_camera_info_schema_and_production_claim_are_canonical() -> None:
    schema = CreateCameraInfo.schema()
    assert schema_from_wire(schema_to_wire(schema)) == schema
    assert schema.outputs[0].type.types == (LOAD3D_CAMERA_TYPE,)
    state = next(item for item in schema.inputs if item.id == "camera_info_state")
    assert state.type.types == (CAMERA_INFO_STATE_TYPE,)
    assert state.socketless and not state.required
    assert schema.input_order == (
        "mode",
        "target_x",
        "target_y",
        "target_z",
        "roll",
        "fov",
        "zoom",
        "camera_type",
        "camera_info_state",
    )
    assert [option.key for option in schema.combos[0].options] == [
        "orbit",
        "look_at",
        "quaternion",
    ]
    assert sum(node is CreateCameraInfo for node in COMFY_NODES) == 1
    manifest = (
        Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-comfy/dinkster-pack.toml"
    )
    assert tomllib.loads(manifest.read_text())["pack"]["namespaces"].count("CreateCameraInfo") == 1


def test_camera_info_discovery_does_not_import_inference_or_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "sys.modules['dinkster_inference'] = None\n"
            "from dinkster_nodes_comfy.camera_info import CreateCameraInfo\n"
            "assert CreateCameraInfo.schema().node_type == 'CreateCameraInfo'\n",
        ],
        check=True,
    )
