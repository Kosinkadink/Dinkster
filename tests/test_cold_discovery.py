"""Canonical public codecs do not load the tensor runtime."""

import subprocess
import sys


def test_canonical_codec_registration_is_torch_and_inference_free() -> None:
    result = subprocess.run(
        [sys.executable, "-c", COLD_DISCOVERY],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


COLD_DISCOVERY = """
import sys

from dinkster_api.v1 import (
    AUDIO_ENCODER_TYPE, CLIP_TYPE, CLIP_VISION_OUTPUT_TYPE, FLOATS_TYPE,
    HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, SAM3_TRACK_DATA_TYPE, TIMESTEPS_RANGE_TYPE,
    TypeRegistry, register_inference_types,
)
from dinkster_inference_wire.sam3_wire import register_sam3_track_type
from dinkster_inference_wire.wan_wire import register_wan_type
from dinkster_values import EncodedLatentTensor
from dinkster_values.builtin_types import SAM3_TRACK_DATA_TYPE as canonical

registry = TypeRegistry()
register_inference_types(registry)
clip = registry.spec(CLIP_TYPE)
register_inference_types(registry)
type_ids = (FLOATS_TYPE, HOOK_KEYFRAMES_TYPE, HOOKS_TYPE, TIMESTEPS_RANGE_TYPE)
assert all(type_id in registry for type_id in type_ids)
assert registry.spec(CLIP_TYPE) is clip
assert clip.coerce is not None

registry = TypeRegistry()
register_inference_types(registry)
register_inference_types(registry)
register_sam3_track_type(registry)
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
with registry.registering_pack('cold-codec-fixture',
                               ('WAN_CAMERA_EMBEDDING', 'TRACKS'), allow_bare=True):
    register_wan_type(registry, 'WAN_CAMERA_EMBEDDING')
    register_wan_type(registry, 'TRACKS')
for name, value in [('WAN_CAMERA_EMBEDDING', tensor), ('TRACKS', {'track_path': tensor}),
                    ('AUDIO_ENCODER_OUTPUT', {'encoded_audio': tensor,
                     'encoded_audio_all_layers': [tensor], 'audio_samples': 0}),
                    ('AUDIO_ENCODER_OUTPUT', {'audio_feature': tensor, 'fps': 15.0})]:
    spec = registry.spec(name)
    data = spec.encode(value)
    spec.validate_encoded(data, {})
    assert spec.decode(data) == value
assert not any(name == 'torch' or name.startswith('torch.')
               or name == 'dinkster_inference' or name.startswith('dinkster_inference.')
               for name in sys.modules)
"""
