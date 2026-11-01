"""The native weld declaration remains usable without a tensor runtime."""

import subprocess
import sys


def test_weld_vertices_cold_schema() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.mesh_bake import BakeAmbientOcclusion
from dinkster_nodes_comfy.mesh_decimate import DecimateMesh
from dinkster_nodes_comfy.mesh_holes import FillHoles
from dinkster_nodes_comfy.mesh_remesh import RemeshMesh
from dinkster_nodes_comfy.mesh_render import RenderMesh
from dinkster_nodes_comfy.mesh_weld import WeldVertices
from dinkster_workers import load_manifest
assert COMFY_NODES.count(WeldVertices) == 1
schema = WeldVertices.schema()
assert [slot.id for slot in schema.inputs] == ['mesh', 'epsilon_rel', 'epsilon_abs']
assert schema.inputs[0].type.types == ('MESH',)
assert schema.outputs[0].type.types == ('MESH',)
assert schema.inputs[1].default == 1e-5 and schema.inputs[2].default == 0.0
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert manifest.namespaces.count('WeldVertices') == 1
assert COMFY_NODES.count(FillHoles) == 1
assert manifest.namespaces.count('FillHoles') == 1
schema = FillHoles.schema()
assert [slot.id for slot in schema.inputs] == [
    'mesh', 'max_perimeter', 'weld_epsilon_rel', 'max_vertices', 'fill_chains'
]
assert [slot.default for slot in schema.inputs[1:]] == [0.03, 1e-5, 16, False]
assert schema.outputs[0].type.types == ('MESH',)
assert COMFY_NODES.count(DecimateMesh) == 1
assert manifest.namespaces.count('DecimateMesh') == 1
schema = DecimateMesh.schema()
assert schema.inputs[1].default == 200_000
assert schema.outputs[0].type.types == ('MESH',)
assert schema.combos[0].id == 'placement_mode'
assert [option.key for option in schema.combos[0].options] == ['midpoint', 'qem']
assert [slot.default for slot in schema.combos[0].options[1].inputs] == [0., 0., 30., True]
assert COMFY_NODES.count(RemeshMesh) == 1
assert manifest.namespaces.count('RemeshMesh') == 1
schema = RemeshMesh.schema()
assert schema.inputs[1].default == 512
assert schema.outputs[0].type.types == ('MESH',)
assert [option.key for option in schema.combos[0].options] == ['udf', 'sdf', 'solid']
assert [slot.default for slot in schema.combos[0].options[0].inputs] == [False, False, False]
assert [slot.default for slot in schema.combos[0].options[1].inputs] == [True, False]
assert [slot.default for slot in schema.combos[0].options[2].inputs] == [20., False]
assert COMFY_NODES.count(RenderMesh) == manifest.namespaces.count('RenderMesh') == 1
schema = RenderMesh.schema()
assert [slot.id for slot in schema.inputs] == [
    'mesh', 'mode', 'width', 'height', 'background', 'model_3d_info', 'camera_info'
]
assert [slot.id for slot in schema.outputs] == ['IMAGE', 'MASK']
assert 'dinkster_nodes_comfy.mesh_render_ops' not in sys.modules
assert COMFY_NODES.count(BakeAmbientOcclusion) == 1
assert manifest.namespaces.count('BakeAmbientOcclusion') == 1
schema = BakeAmbientOcclusion.schema()
assert [slot.id for slot in schema.inputs] == [
    'low_poly', 'high_poly', 'resolution', 'samples', 'max_distance', 'strength', 'bias'
]
assert schema.outputs[0].type.types == ('IMAGE',)
assert 'dinkster_nodes_comfy.mesh_bake_ops' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
