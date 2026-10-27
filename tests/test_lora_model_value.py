"""Cold contracts for the portable public LORA_MODEL value type."""

from __future__ import annotations

import subprocess
import sys
from collections import OrderedDict

import pytest
from dinkster_api.v1 import LATENT_TYPE, LORA_MODEL_TYPE, TypeRegistry, register_inference_types
from dinkster_values import EncodedLatentTensor, decode_latent, encode_latent, latent_fingerprint


def _value() -> OrderedDict[str, EncodedLatentTensor]:
    return OrderedDict(
        (
            ("lora_te.alpha", EncodedLatentTensor("float32", (), b"\x00\x00\x80?")),
            (
                "lora_unet.down.weight",
                EncodedLatentTensor("float16", (2, 3), b"\x00<\x00@\x00B\x00D\x00E\x00F"),
            ),
            ("lora_unet.up.weight", EncodedLatentTensor("bfloat16", (1, 0, 4), b"")),
        ),
    )


def test_lora_model_roundtrip_preserves_key_order_and_tensor_records() -> None:
    registry = TypeRegistry()
    registered = register_inference_types(registry)
    spec = registry.spec(LORA_MODEL_TYPE)

    assert LORA_MODEL_TYPE == "LORA_MODEL"
    assert spec in registered
    assert register_inference_types(registry) == registered

    source = _value()
    encoded = spec.encode(source)
    restored = spec.decode(encoded)

    assert type(restored) is dict
    assert tuple(restored) == tuple(source)
    assert restored == source
    assert spec.fingerprint is not None
    assert spec.fingerprint(source) == spec.fingerprint(restored)
    assert spec.fingerprint(source) != spec.fingerprint(dict(reversed(tuple(source.items()))))
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, {})
    assert spec.decode(spec.encode({})) == {}


def test_default_latent_rejects_ordered_mapping_frames_and_keeps_existing_identity() -> None:
    value = {"samples": EncodedLatentTensor("float32", (1,), b"\x00\x00\x80?")}
    encoded = encode_latent(value)

    assert encode_latent(decode_latent(encoded)) == encoded
    assert latent_fingerprint(LATENT_TYPE)(value) == "ab5ebb24357a9feccebd5d0e2f3717711d2b6ddf"

    registry = TypeRegistry()
    register_inference_types(registry)
    ordered_frame = registry.spec(LORA_MODEL_TYPE).encode(_value())
    with pytest.raises(ValueError, match="invalid mapping"):
        decode_latent(ordered_frame)


def test_lora_model_registration_remains_torch_and_runtime_free() -> None:
    script = """
import sys
from dinkster_api.v1 import LORA_MODEL_TYPE, TypeRegistry, register_inference_types
from dinkster_values import EncodedLatentTensor

registry = TypeRegistry()
registered = register_inference_types(registry)
assert register_inference_types(registry) == registered
value = {
    'second': EncodedLatentTensor('float16', (0,), b''),
    'first': EncodedLatentTensor('float32', (), b'\\x00\\x00\\x80?'),
}
spec = registry.spec(LORA_MODEL_TYPE)
encoded = spec.encode(value)
spec.validate_encoded(encoded, {})
assert tuple(spec.decode(encoded)) == ('second', 'first')
assert not any(
    name == 'torch' or name.startswith('torch.') or name == 'dinkster_inference'
    or name.startswith('dinkster_inference.') for name in sys.modules
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
