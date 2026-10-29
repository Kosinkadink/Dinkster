"""The complete manifest preserves the public node registry's order."""

from dinkster_nodes_comfy import COMFY_NODES
from dinkster_workers import load_manifest


def test_manifest_claims_match_the_complete_registry() -> None:
    manifest = load_manifest("packages/dinkster-nodes-comfy/dinkster-pack.toml")
    assert ("devices", *(node.schema().node_type for node in COMFY_NODES)) == manifest.namespaces
