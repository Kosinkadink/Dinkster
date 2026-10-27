"""Portable IC-LoRA parameter mapping contracts."""

from __future__ import annotations

import subprocess
import sys

import pytest
from dinkster_api.v1 import IC_LORA_PARAMETERS_TYPE, TypeRegistry, register_inference_types
from dinkster_values import EncodedLatentTensor, encode_latent


def test_ic_lora_parameters_roundtrip_preserves_mapping_order_and_extras() -> None:
    value = {
        "reference_downscale_factor": 2,
        "nested": {"keep": ("source", 3)},
        "tensor": EncodedLatentTensor("float16", (1, 2), b"\x00\x3c\x00\xc0"),
    }
    registry = TypeRegistry()
    registered = register_inference_types(registry)
    spec = registry.spec(IC_LORA_PARAMETERS_TYPE)

    assert spec in registered
    assert register_inference_types(registry) == registered
    encoded = spec.encode(value)
    assert spec.decode(encoded) == value
    assert list(spec.decode(encoded)) == list(value)
    assert spec.fingerprint is not None
    assert spec.fingerprint(value) == spec.fingerprint(spec.decode(encoded))
    assert spec.fingerprint(value) != spec.fingerprint({**value, "extra": True})


def test_ic_lora_parameters_rejects_non_mapping_and_invalid_frames() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(IC_LORA_PARAMETERS_TYPE)
    assert spec.validate_encoded is not None

    with pytest.raises(ValueError):
        spec.encode(("reference_downscale_factor", 2))
    payload = encode_latent({"samples": {"reference_downscale_factor": 1}, "extra": 1})
    with pytest.raises(ValueError):
        spec.decode(payload)
    with pytest.raises(ValueError):
        spec.validate_encoded(payload, {})


def test_ic_lora_parameters_public_registration_is_cold_and_idempotent() -> None:
    script = "\n".join(
        (
            "import sys",
            "from dinkster_api.v1 import (",
            "    IC_LORA_PARAMETERS_TYPE, TypeRegistry, register_inference_types,",
            ")",
            "registry = TypeRegistry()",
            "first = register_inference_types(registry)",
            "second = register_inference_types(registry)",
            "assert first == second",
            "spec = registry.spec(IC_LORA_PARAMETERS_TYPE)",
            "value = {'reference_downscale_factor': 2, 'extra': {'preserve': True}}",
            "assert spec.decode(spec.encode(value)) == value",
            "blocked = ('torch', 'dinkster_inference')",
            "assert not any(name.split('.')[0] in blocked for name in sys.modules)",
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        check=True,
    )
