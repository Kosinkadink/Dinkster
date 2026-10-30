"""Mesh merge declarations remain available without tensor runtimes."""

import subprocess
import sys


def test_merge_meshes_cold_schema_registration_and_family_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_api.v1 import MESH_TYPE, schema_from_wire, schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.mesh import MergeMeshes
from dinkster_workers import load_manifest
schema = schema_from_wire(schema_to_wire(MergeMeshes.schema()))
assert schema.node_type == 'MergeMeshes'
assert schema.display_name == 'Merge Meshes' and schema.category == '3d/mesh'
family, = schema.input_families
assert family.id == 'meshes' and family.member_prefix == 'mesh'
assert family.min_members == 2 and family.max_members == 50
assert family.optional_after_minimum
assert family.template[0].type.types == (MESH_TYPE,)
assert [o.id for o in schema.outputs] == ['mesh']
assert COMFY_NODES.count(MergeMeshes) == 1
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert manifest.namespaces.count('MergeMeshes') == 1
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
