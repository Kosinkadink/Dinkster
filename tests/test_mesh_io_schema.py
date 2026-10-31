"""The mesh file declaration and registration do not import tensor runtimes."""

import subprocess
import sys


def test_get_3d_components_cold_schema() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['torch'] = None
sys.modules['dinkster_inference'] = None
from dinkster_nodes_comfy import COMFY_NODES
from dinkster_nodes_comfy.mesh_io import Get3DComponents, MeshToFile3D, SaveGLB
from dinkster_nodes_comfy.save3d import Save3DAdvanced, SaveGaussianSplat, SavePointCloud
from dinkster_workers import load_manifest
assert COMFY_NODES.count(Get3DComponents) == 1
schema = Get3DComponents.schema()
assert [slot.id for slot in schema.inputs] == ['model_3d']
assert set(schema.inputs[0].type.types) == {
    'FILE_3D', 'FILE_3D_GLB', 'FILE_3D_GLTF', 'FILE_3D_OBJ', 'FILE_3D_STL'
}
assert [(slot.id, slot.type.types) for slot in schema.outputs] == [('mesh', ('MESH',))]
manifest = load_manifest('packages/dinkster-nodes-comfy/dinkster-pack.toml')
assert manifest.namespaces.count('Get3DComponents') == 1
assert COMFY_NODES.count(MeshToFile3D) == 1
assert manifest.namespaces.count('MeshToFile3D') == 1
schema = MeshToFile3D.schema()
assert [(slot.id, slot.type.types) for slot in schema.inputs] == [('mesh', ('MESH',))]
assert [(slot.id, slot.type.types) for slot in schema.outputs] == [('model_3d', ('FILE_3D_GLB',))]
assert COMFY_NODES.count(SaveGLB) == manifest.namespaces.count('SaveGLB') == 1
schema = SaveGLB.schema()
assert [slot.id for slot in schema.inputs] == ['mesh', 'filename_prefix', 'mount']
assert schema.inputs[-1].adjacent_to == 'filename_prefix'
assert schema.outputs == () and schema.output_node and not schema.idempotent
for node in (Save3DAdvanced, SaveGaussianSplat, SavePointCloud):
    assert COMFY_NODES.count(node) == manifest.namespaces.count(node.__name__) == 1
    schema = node.schema()
    assert [slot.id for slot in schema.inputs] == [
        'model_3d', 'filename_prefix', 'viewport_state', 'model_3d_info',
        'camera_info', 'width', 'height', 'mount'
    ]
    assert schema.inputs[-1].adjacent_to == 'filename_prefix'
    assert [slot.id for slot in schema.outputs] == [
        'model_3d', 'model_3d_info', 'camera_info', 'width', 'height'
    ]
    assert schema.output_node and not schema.idempotent
assert 'dinkster_nodes_comfy.mesh_gltf_read' not in sys.modules
assert 'dinkster_nodes_comfy.mesh_file_read' not in sys.modules
assert 'dinkster_native.model3d.mesh' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
