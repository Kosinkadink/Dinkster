from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
import torch
from dinkster_inference import (
    FLOAT32,
    TAESDConfig,
    TAESDMemoryEstimator,
    TensorGeometry,
    taesd_layout,
)
from dinkster_inference_torch import TAESD, TAESDDecoder, TAESDEncoder, taesd_codec_plugin


def test_architecture_state_dict_matches_reference_listing() -> None:
    assert {k: tuple(v.shape) for k, v in TAESDEncoder().state_dict().items()} == taesd_layout(
        "encoder"
    )
    assert {k: tuple(v.shape) for k, v in TAESDDecoder().state_dict().items()} == taesd_layout(
        "decoder"
    )


def test_strict_load_and_transforms() -> None:
    encoder, decoder = TAESDEncoder(), TAESDDecoder()
    encoder.load_state_dict(encoder.state_dict(), strict=True, assign=True)
    decoder.load_state_dict(decoder.state_dict(), strict=True, assign=True)
    model = TAESD(TAESDConfig("sd15", "encoder"), encoder, decoder)
    plugin = taesd_codec_plugin(model)
    seen: list[torch.Tensor] = []
    encoder.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    encoded = plugin.encode(torch.zeros(1, 3, 8, 8))
    # ComfyUI's outer VAE process_input and TAESD.encode mapping cancel.
    assert torch.equal(seen[0], torch.zeros(1, 3, 8, 8))
    assert encoded.shape == (1, 4, 1, 1)
    assert plugin.decode(encoded).shape == (1, 3, 8, 8)


def test_missing_half_and_wrong_latent_width_refuse_loudly() -> None:
    config = TAESDConfig("sdxl", "decoder")
    plugin = taesd_codec_plugin(TAESD(config, TAESDEncoder(), TAESDDecoder()))
    with pytest.raises(ValueError, match="latent width 4"):
        plugin.decode(torch.zeros(1, 8, 1, 1))


def test_codec_crops_encode_and_uses_pinned_memory_estimates() -> None:
    plugin = taesd_codec_plugin(
        TAESD(TAESDConfig("sd15", "encoder"), TAESDEncoder(), TAESDDecoder())
    )
    assert plugin.encode(torch.zeros(1, 3, 17, 19)).shape == (1, 4, 2, 2)
    memory = TAESDMemoryEstimator()
    assert memory.encode_bytes(TensorGeometry((1, 3, 16, 24), FLOAT32)) == 1767 * 16 * 24 * 4
    assert memory.decode_bytes(TensorGeometry((1, 4, 2, 3), FLOAT32)) == 2178 * 2 * 3 * 64 * 4


@pytest.mark.parametrize(("family", "stem"), (("sd15", "taesd"), ("sdxl", "taesdxl")))
def test_official_weights_replay_comfyui_golden(family: Literal["sd15", "sdxl"], stem: str) -> None:
    weights = Path("/tmp/dinkster-taesd-weights")
    if not weights.exists():
        pytest.skip("official TAESD weights are not installed")
    golden = json.loads((Path(__file__).parent / "goldens/taesd_goldens.json").read_text())
    assert golden["_meta"]["reference_commit"] == ("b78cec879b9460d5cb25228a83a942fb78d2cd24")
    encoder, decoder = TAESDEncoder(), TAESDDecoder()
    encoder.load_state_dict(
        torch.load(weights / f"{stem}_encoder.pth", map_location="cpu", weights_only=True),
        strict=True,
        assign=True,
    )
    decoder.load_state_dict(
        torch.load(weights / f"{stem}_decoder.pth", map_location="cpu", weights_only=True),
        strict=True,
        assign=True,
    )
    plugin = taesd_codec_plugin(TAESD(TAESDConfig(family, "encoder"), encoder, decoder))
    content = torch.linspace(0.0, 1.0, 16 * 16 * 3).reshape(1, 16, 16, 3).movedim(-1, 1)
    latent = plugin.encode(content)
    decoded = plugin.decode(latent)
    for name, tensor in (("encode", latent), ("decode", decoded.movedim(1, -1))):
        expected = golden[family][name]
        assert list(tensor.shape) == expected["shape"]
        assert float(tensor.detach().sum()) == pytest.approx(expected["sum"], abs=5e-5)
        flat = tensor.detach().flatten()
        for index, value in expected["samples"]:
            assert float(flat[index]) == pytest.approx(value, abs=1e-6)
