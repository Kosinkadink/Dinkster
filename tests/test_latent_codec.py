from __future__ import annotations

import ctypes
import json

import pytest
from dinkster_values import (
    LATENT_CODEC_MAGIC,
    EncodedLatentTensor,
    EncodedMultiStreamLatent,
    decode_latent,
    encode_latent,
    validate_latent_encoded,
)
from dinkster_values.latent_codec import _copy_address


def _tensor(*, fill: int, shape: tuple[int, ...]) -> EncodedLatentTensor:
    size = 1
    for dimension in shape:
        size *= dimension
    return EncodedLatentTensor("float32", shape, bytes((fill, 0, 0, 0)) * size)


def test_tensor_buffer_copy_uses_pointer_sized_lengths() -> None:
    source = ctypes.create_string_buffer(b"latent")
    assert _copy_address(ctypes.addressof(source), 6) == b"latent"
    assert _copy_address(0, 0) == b""
    assert (ctypes.c_ubyte * (2**32 + 2))._length_ == 2**32 + 2


def test_structural_latent_codec_is_deterministic_and_round_trips_streams() -> None:
    video = _tensor(fill=1, shape=(1, 2, 3))
    audio = _tensor(fill=2, shape=(1, 4))
    value = {
        "samples": EncodedMultiStreamLatent((("video", video), ("audio", audio))),
        "noise_mask": EncodedMultiStreamLatent(
            (("video", _tensor(fill=3, shape=(1, 1, 3))), ("audio", audio))
        ),
        "batch_index": (0,),
        "metadata": {"name": "value", "enabled": True},
    }
    encoded = encode_latent(value)
    reordered = encode_latent(dict(reversed(tuple(value.items()))))
    assert encoded == reordered
    assert encoded.startswith(LATENT_CODEC_MAGIC)
    validate_latent_encoded(encoded, {})

    decoded = decode_latent(encoded)
    assert isinstance(decoded, dict)
    samples = decoded["samples"]
    assert type(samples) is EncodedMultiStreamLatent
    assert samples.roles == ("video", "audio")
    assert samples.streams[0][1] == video
    assert decoded["batch_index"] == (0,)
    assert decoded["metadata"] == {"enabled": True, "name": "value"}


@pytest.mark.parametrize("reserved", ("tensor", "multi", "tuple", "map", "user"))
def test_latent_codec_escapes_reserved_mapping_keys(reserved: str) -> None:
    value = {
        "samples": _tensor(fill=1, shape=(1,)),
        "metadata": {"$": reserved, "items": {"$": "tensor"}},
    }
    assert decode_latent(encode_latent(value)) == value


def test_latent_codec_validator_rejects_old_and_corrupt_frames() -> None:
    with pytest.raises(ValueError, match="framing"):
        validate_latent_encoded(b"pickle:not-a-latent", {})
    encoded = encode_latent({"samples": _tensor(fill=1, shape=(1,))})
    with pytest.raises(ValueError, match="payload"):
        validate_latent_encoded(encoded[:-4], {})
    with pytest.raises(ValueError, match="dtype and shape"):
        validate_latent_encoded(encoded.replace(b"float32", b"float99"), {})

    prefix = LATENT_CODEC_MAGIC + b"\x01"
    malformed = {"$": "map", "items": {}, "extra": True}
    with pytest.raises(ValueError, match="mapping"):
        validate_latent_encoded(prefix + json.dumps(malformed).encode(), {})
