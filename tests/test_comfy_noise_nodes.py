"""Public native noise schemas and manifest remain cold-process safe."""

from dinkster_api.v1 import NumberWidget
from dinkster_sampling import NODES
from dinkster_sampling.noise import AddNoise, DisableNoise, RandomNoise
from dinkster_workers import load_manifest


def test_noise_nodes_publish_the_fixed_core_interfaces_and_manifest_entries() -> None:
    nodes = (RandomNoise, DisableNoise, AddNoise)
    assert all(node in NODES for node in nodes)
    assert tuple(node.schema().node_type for node in nodes) == (
        "dinkster.sampling.random_noise",
        "dinkster.sampling.disable_noise",
        "dinkster.sampling.add_noise",
    )
    seed = RandomNoise.schema().inputs[0]
    assert seed.default == 0
    assert seed.widget == NumberWidget(
        min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate="fixed"
    )
    manifest = load_manifest("packages/dinkster-nodes-sampling/dinkster-pack.toml")
    assert manifest.namespaces == ("dinkster.sampling",)
