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
from dinkster_3d import NODES
from dinkster_3d.mesh_io import Get3DComponents, MeshToFile3D, SaveGLB
from dinkster_3d.save3d import Save3DAdvanced, SaveGaussianSplat, SavePointCloud
from dinkster_3d.preview3d import (
    Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud,
)
from dinkster_3d.splat_files import File3DToSplat, SplatToFile3D
from dinkster_workers import load_manifest
assert NODES.count(Get3DComponents) == 1
schema = Get3DComponents.schema()
assert [slot.id for slot in schema.inputs] == ['model_3d']
assert set(schema.inputs[0].type.types) == {
    'FILE_3D', 'FILE_3D_GLB', 'FILE_3D_GLTF', 'FILE_3D_OBJ', 'FILE_3D_STL'
}
assert [(slot.id, slot.type.types) for slot in schema.outputs] == [('mesh', ('MESH',))]
manifest = load_manifest('packages/dinkster-nodes-3d/dinkster-pack.toml')
assert 'dinkster.3d' in manifest.namespaces
assert NODES.count(MeshToFile3D) == 1
schema = MeshToFile3D.schema()
assert [(slot.id, slot.type.types) for slot in schema.inputs] == [('mesh', ('MESH',))]
assert [(slot.id, slot.type.types) for slot in schema.outputs] == [('model_3d', ('FILE_3D_GLB',))]
assert NODES.count(SaveGLB) == 1
schema = SaveGLB.schema()
assert [slot.id for slot in schema.inputs] == ['mesh', 'filename_prefix', 'mount']
assert schema.inputs[-1].adjacent_to == 'filename_prefix'
assert schema.outputs == () and schema.output_node and not schema.idempotent
for node in (Save3DAdvanced, SaveGaussianSplat, SavePointCloud):
    assert NODES.count(node) == 1 and node.schema().node_type.startswith("dinkster.3d.")
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
assert NODES.count(SplatToFile3D) == 1
schema = SplatToFile3D.schema()
assert [slot.id for slot in schema.inputs] == ['splat', 'format']
assert schema.inputs[1].widget.options == ('ply', 'ksplat', 'spz')
assert [(slot.id, slot.type.types) for slot in schema.outputs] == [
    ('model_3d', ('FILE_3D_SPLAT_ANY',))
]
assert NODES.count(File3DToSplat) == 1
schema = File3DToSplat.schema()
assert [slot.id for slot in schema.inputs] == ['model_3d']
assert set(schema.inputs[0].type.types) == {
    'FILE_3D', 'FILE_3D_SPLAT_ANY', 'FILE_3D_PLY', 'FILE_3D_SPLAT', 'FILE_3D_KSPLAT', 'FILE_3D_SPZ'
}
assert [(slot.id, slot.type.types) for slot in schema.outputs] == [('splat', ('SPLAT',))]
assert 'dinkster_3d.mesh_gltf_read' not in sys.modules
assert 'dinkster_3d.mesh_file_read' not in sys.modules
assert 'dinkster_native.model3d.mesh' not in sys.modules
for node in (Preview3DAdvanced, PreviewGaussianSplat, PreviewPointCloud):
    assert NODES.count(node) == 1 and node.schema().node_type.startswith("dinkster.3d.")
    schema = node.schema()
    assert [slot.id for slot in schema.inputs] == [
        'model_3d', 'model_3d_info', 'viewport_state', 'camera_info', 'width', 'height'
    ]
    assert [slot.id for slot in schema.outputs] == [
        'model_3d', 'model_3d_info', 'camera_info', 'width', 'height'
    ]
    assert schema.output_node and schema.outputs[0].preview
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
