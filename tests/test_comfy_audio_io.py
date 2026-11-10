"""Audio file nodes are discoverable without importing the tensor runtime."""

import subprocess
import sys

from dinkster_api.v1 import ASSET_TYPE, AUDIO_TYPE, schema_from_wire, schema_to_wire
from dinkster_audio import NODES
from dinkster_audio.audio_io import LoadAudio, PreviewAudio
from dinkster_workers import load_manifest


def test_native_audio_io_schemas_and_manifest() -> None:
    schema = LoadAudio.schema()
    assert [(item.id, item.type.types) for item in schema.inputs] == [("audio", (ASSET_TYPE,))]
    assert schema.inputs[0].source_filename is not None
    assert schema.inputs[0].source_filename.kind == "media/audio"
    assert schema.inputs[0].source_filename.category == "input"
    assert [(item.id, item.type.types) for item in schema.outputs] == [("AUDIO", (AUDIO_TYPE,))]
    assert PreviewAudio.schema().output_node
    for node in (LoadAudio, PreviewAudio):
        assert schema_from_wire(schema_to_wire(node.schema())) == node.schema()
        assert node in NODES
    manifest = load_manifest("packages/dinkster-nodes-audio/dinkster-pack.toml")
    assert manifest.namespaces == ("dinkster.audio",)
    assert all(node.schema().node_type.startswith("dinkster.audio.") for node in NODES)


def test_native_audio_io_cold_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch']=None; sys.modules['dinkster_inference']=None; "
            "from dinkster_audio import NODES; "
            "assert {'dinkster.audio.load_audio','dinkster.audio.preview_audio'} "
            "<= {n.schema().node_type for n in NODES}; "
            "assert not any(n.startswith('dinkster_inference.') for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
