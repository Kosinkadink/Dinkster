"""CLIP vision attributes survive host transport without a model runtime."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
from dinkster_api.v1 import CLIP_VISION_OUTPUT_TYPE, TypeRegistry, register_inference_types
from dinkster_values import EncodedLatentTensor, encode_latent


def test_vision_output_preserves_all_attributes_and_cache_identity() -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    register_inference_types(registry)
    spec = registry.spec(CLIP_VISION_OUTPUT_TYPE)
    half = EncodedLatentTensor("float16", (1, 2), b"\x00\x3c\x00\xc0")
    bfloat = EncodedLatentTensor("bfloat16", (1, 1), b"\x80\x3f")
    source = SimpleNamespace(
        last_hidden_state=half,
        image_embeds=bfloat,
        penultimate_hidden_states=EncodedLatentTensor("float32", (0, 3), b""),
        image_sizes=[(224, 336, 3), (448, 224, 3)],
        mm_projected=None,
        all_hidden_states=(half, {"layer": 2, "selected": True, "label": "last", "$": "map"}),
    )
    before = vars(source).copy()
    encoded = spec.encode(source)
    assert spec.validate_encoded is not None
    spec.validate_encoded(encoded, {})
    decoded = spec.decode(encoded)
    assert decoded == {
        "last_hidden_state": half,
        "image_embeds": bfloat,
        "penultimate_hidden_states": EncodedLatentTensor("float32", (0, 3), b""),
        "image_sizes": [(224, 336, 3), (448, 224, 3)],
        "mm_projected": None,
        "all_hidden_states": (
            half,
            {"layer": 2, "selected": True, "label": "last", "$": "map"},
        ),
    }
    assert vars(source) == before
    assert spec.fingerprint is not None
    identity = spec.fingerprint(source)
    assert identity == spec.fingerprint(decoded)
    assert identity == spec.fingerprint(dict(reversed(tuple(before.items()))))
    without_optional = {key: value for key, value in before.items() if key != "mm_projected"}
    decoded_without_optional = spec.decode(spec.encode(without_optional))
    assert isinstance(decoded_without_optional, dict)
    assert "mm_projected" not in decoded_without_optional
    assert spec.fingerprint(without_optional) != identity
    assert spec.fingerprint({**before, "image_sizes": [(336, 224, 3)]}) != identity
    assert spec.fingerprint({**before, "image_embeds": half}) != identity


@pytest.mark.parametrize(
    "data",
    (
        b"pickle:not-a-vision-output",
        encode_latent({"samples": 4}),
        encode_latent({"samples": {}, "extra": 1}),
        encode_latent({"samples": {"x": EncodedLatentTensor("float16", (2,), b"\x00\x00")}}),
    ),
)
def test_vision_output_rejects_malformed_payloads(data: bytes) -> None:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(CLIP_VISION_OUTPUT_TYPE)
    assert spec.validate_encoded is not None
    with pytest.raises(ValueError):
        spec.validate_encoded(data, {})
    with pytest.raises(ValueError):
        spec.decode(data)


def test_vision_registration_and_decode_do_not_import_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from dinkster_api.v1 import CLIP_VISION_OUTPUT_TYPE, TypeRegistry, "
            "register_inference_types\n"
            "r = TypeRegistry()\n"
            "register_inference_types(r)\n"
            "s = r.spec(CLIP_VISION_OUTPUT_TYPE)\n"
            "b = s.encode({'image_sizes': [(2, 3, 3)], 'mm_projected': None})\n"
            "s.validate_encoded(b, {})\n"
            "assert s.decode(b) == {'image_sizes': [(2, 3, 3)], 'mm_projected': None}\n"
            "assert 'torch' not in sys.modules\n"
            "assert not any(k.startswith('dinkster_inference.') for k in sys.modules)\n",
        ],
        check=True,
    )
