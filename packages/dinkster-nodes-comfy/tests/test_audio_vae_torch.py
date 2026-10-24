"""CPU behavior for native ComfyUI audio VAE nodes."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from dinkster_nodes_comfy.audio_vae import VAEDecodeAudio, VAEDecodeAudioTiled, VAEEncodeAudio


class AudioVAE:
    audio_sample_rate = 24000
    audio_sample_rate_output = 32000

    def __init__(self) -> None:
        self.encoded: Any = None
        self.decoded: Any = None
        self.tiled: tuple[Any, int, int, int | None] | None = None

    def encode(self, waveform: Any) -> Any:
        self.encoded = waveform
        return waveform + 2

    def decode(self, latent: Any) -> Any:
        self.decoded = latent
        return latent.movedim(1, -1)

    def decode_tiled(self, latent: Any, *, tile_x: int, tile_y: int, overlap: int | None) -> Any:
        self.tiled = (latent, tile_x, tile_y, overlap)
        return latent.movedim(1, -1)


def test_encode_preserves_layout_and_rejects_no_audio() -> None:
    vae = AudioVAE()
    waveform = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    source = {"waveform": waveform, "sample_rate": 24000, "extra": "source"}
    encoded = VAEEncodeAudio.execute(audio=source, vae=vae)["LATENT"]
    assert torch.equal(vae.encoded, waveform.movedim(1, -1))
    assert torch.equal(encoded["samples"], waveform.movedim(1, -1) + 2)
    assert encoded == {"samples": encoded["samples"]}
    assert source["waveform"] is waveform and source["extra"] == "source"
    with pytest.raises(ValueError, match="VAEEncodeAudio: input audio is None"):
        VAEEncodeAudio.execute(audio=None, vae=vae)


def test_decode_uses_last_nested_stream_normalizes_each_batch_and_selects_rate() -> None:
    vae = AudioVAE()
    first = torch.full((1, 2, 3), 9.0)
    last = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]])
    expected = last.clone()
    std = expected.std(dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    nested = torch.nested.nested_tensor([first, last])
    samples = {"samples": nested, "sample_rate": 48000, "extra": "source"}
    output = VAEDecodeAudio.execute(samples=samples, vae=vae)["AUDIO"]
    assert torch.equal(vae.decoded, expected / std)
    torch.testing.assert_close(output["waveform"], expected / std)
    assert output["sample_rate"] == 48000
    assert output["waveform"].dtype == last.dtype and output["waveform"].device == last.device
    assert samples["extra"] == "source"


def test_decode_rate_falls_back_to_vae_output_rate() -> None:
    samples = {"samples": torch.ones(1, 2, 3)}
    output = VAEDecodeAudio.execute(samples=samples, vae=AudioVAE())["AUDIO"]
    assert output["sample_rate"] == 32000

    class InputRateVAE(AudioVAE):
        audio_sample_rate_output = None

    output = VAEDecodeAudio.execute(samples=samples, vae=InputRateVAE())["AUDIO"]
    assert output["sample_rate"] is None


def test_tiled_decode_delegates_exact_arguments() -> None:
    vae = AudioVAE()
    samples = {"samples": torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)}
    output = VAEDecodeAudioTiled.execute(samples=samples, vae=vae, tile_size=520, overlap=72)[
        "AUDIO"
    ]
    assert vae.tiled is not None
    latent, tile_x, tile_y, overlap = vae.tiled
    assert torch.equal(latent, samples["samples"])
    assert (tile_x, tile_y, overlap) == (520, 520, 72)
    assert output["sample_rate"] == 32000
