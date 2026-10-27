"""Public native noise schemas and manifest remain cold-process safe."""

import subprocess
import sys

from dinkster_api.v1 import NumberWidget
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.noise import NOISE_NODES, AddNoise, DisableNoise, RandomNoise
from dinkster_workers import load_manifest


def test_noise_nodes_publish_the_fixed_core_interfaces_and_manifest_entries() -> None:
    assert NOISE_NODES == (RandomNoise, DisableNoise, AddNoise)
    assert all(node in COMFY_NODES for node in NOISE_NODES)
    assert tuple(node.schema().node_type for node in NOISE_NODES) == (
        "RandomNoise",
        "DisableNoise",
        "AddNoise",
    )
    seed = RandomNoise.schema().inputs[0]
    assert seed.default == 0
    assert seed.widget == NumberWidget(
        min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate="fixed"
    )
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert tuple(
        name for name in manifest.namespaces if name in {"RandomNoise", "DisableNoise", "AddNoise"}
    ) == (
        "RandomNoise",
        "DisableNoise",
        "AddNoise",
    )


def test_noise_schemas_are_torch_and_inference_free_in_a_cold_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["torch"] = None
sys.modules["dinkster_inference"] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.noise import NOISE_NODES
assert all(node in COMFY_NODES for node in NOISE_NODES)
for node in NOISE_NODES:
    node.schema()
assert "torch" not in sys.modules or sys.modules["torch"] is None
assert "dinkster_inference" not in sys.modules or sys.modules["dinkster_inference"] is None
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
