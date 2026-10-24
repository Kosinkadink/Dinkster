"""Public structural latent identities across portable and live payloads."""

import pytest
import torch
from dinkster_api.v1 import LatentStream, MultiStreamLatent, materialize_latent
from dinkster_values import decode_latent, encode_latent


@pytest.mark.parametrize("roles", (("video", "audio"), ("audio", "video")))
def test_public_multi_stream_roundtrip_preserves_order_and_payloads(
    roles: tuple[str, str],
) -> None:
    video = torch.arange(240, dtype=torch.float64).reshape(2, 3, 4, 2, 5) / 7
    audio = torch.arange(84, dtype=torch.float32).reshape(2, 3, 2, 7) - 19
    payloads = {"video": video, "audio": audio}
    source = MultiStreamLatent.from_pairs((role, payloads[role]) for role in roles)
    assert materialize_latent(source) is source
    assert all(stream.payload is payloads[stream.role] for stream in source.streams)

    restored = materialize_latent(decode_latent(encode_latent({"samples": source})))
    streams = restored["samples"]
    assert type(streams) is MultiStreamLatent
    assert streams.roles == roles
    for stream in streams.streams:
        assert type(stream) is LatentStream
        expected = payloads[stream.role]
        assert stream.payload.dtype == expected.dtype
        assert stream.payload.shape == expected.shape
        assert stream.payload.device.type == "cpu"
        assert torch.equal(stream.payload, expected)
        assert source.by_role(stream.role) is expected

    plain = {"samples": video}
    assert materialize_latent(plain) is plain
    decoded_plain = materialize_latent(decode_latent(encode_latent(plain)))
    assert type(decoded_plain["samples"]) is torch.Tensor
    assert decoded_plain["samples"].dtype == video.dtype
    assert torch.equal(decoded_plain["samples"], video)
