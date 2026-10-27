"""Schema discovery and public codecs do not load the tensor runtime."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE_MODULES = tuple(
    f"dinkster_nodes_comfy.{path.stem}"
    for path in sorted(
        (ROOT / "packages/dinkster-nodes-comfy/src/dinkster_nodes_comfy").glob("*.py")
    )
    if path.stem != "__init__"
)


@pytest.fixture(scope="module")
def cold_modules() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-c", COLD_DISCOVERY, json.dumps(NODE_MODULES)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout))


@pytest.mark.parametrize("module", NODE_MODULES)
def test_node_module_discovery_is_torch_and_inference_free(
    module: str, cold_modules: set[str]
) -> None:
    assert module in cold_modules


COLD_DISCOVERY = """
import importlib
import json
import sys

from dinkster_schema import schema_to_wire
from dinkster_nodes_comfy import COMFY_NODES

modules = json.loads(sys.argv[1])
for name in modules:
    module = importlib.import_module(name)
    for node in COMFY_NODES:
        if node.__module__ == name:
            assert schema_to_wire(node.schema()), (name, node)
    forbidden = [name for name in sys.modules if name == 'torch' or name.startswith('torch.')
                 or name == 'dinkster_inference' or name.startswith('dinkster_inference.')]
    assert not forbidden, (module.__name__, forbidden)

from dinkster_api.v1 import (
    AUDIO_ENCODER_TYPE, CLIP_TYPE, CLIP_VISION_OUTPUT_TYPE, FLOATS_TYPE,
    HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, SAM3_TRACK_DATA_TYPE, TIMESTEPS_RANGE_TYPE,
    TypeRegistry, register_inference_types,
)
from dinkster_native.native import register_native_types
from dinkster_values import EncodedLatentTensor
from dinkster_values.builtin_types import SAM3_TRACK_DATA_TYPE as canonical

for first, second in ((register_native_types, register_inference_types),
                      (register_inference_types, register_native_types)):
    registry = TypeRegistry()
    first(registry)
    clip = registry.spec(CLIP_TYPE)
    second(registry)
    type_ids = (FLOATS_TYPE, HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, TIMESTEPS_RANGE_TYPE)
    assert all(type_id in registry for type_id in type_ids)
    assert registry.spec(CLIP_TYPE) is clip
    assert clip.coerce is not None

registry = TypeRegistry()
register_inference_types(registry)
register_inference_types(registry)
assert AUDIO_ENCODER_TYPE in registry
assert SAM3_TRACK_DATA_TYPE is canonical
spec = registry.spec(canonical)
value = {'packed_masks': None, 'orig_size': (2, 5), 'n_frames': 0}
data = spec.encode(value)
spec.validate_encoded(data, {})
assert spec.decode(data) == value
spec = registry.spec(CLIP_VISION_OUTPUT_TYPE)
value = {'image_sizes': [(2, 3, 3)], 'mm_projected': None}
data = spec.encode(value)
spec.validate_encoded(data, {})
assert spec.decode(data) == value
tensor = EncodedLatentTensor('bool', (0, 3), b'')
for name, value in [('WAN_CAMERA_EMBEDDING', tensor), ('TRACKS', {'track_path': tensor}),
                    ('AUDIO_ENCODER_OUTPUT', {'encoded_audio': tensor,
                     'encoded_audio_all_layers': [tensor], 'audio_samples': 0})]:
    spec = registry.spec(name)
    data = spec.encode(value)
    spec.validate_encoded(data, {})
    assert spec.decode(data) == value
assert not any(name == 'torch' or name.startswith('torch.')
               or name == 'dinkster_inference' or name.startswith('dinkster_inference.')
               for name in sys.modules)
print(json.dumps(modules))
"""
