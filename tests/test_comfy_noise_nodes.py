"""Public native noise schemas and manifest remain cold-process safe."""

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
